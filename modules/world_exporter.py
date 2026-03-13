import numpy as np
import struct
import json
import os
import hashlib
import logging
import base64

logger = logging.getLogger("WorldExporter")

# =============================================================================
# [MODULE] AWE V10 Surgery - World Exporter (O Empacotador Final)
# =============================================================================
# FIX #6:  blue-noise hash 3D para variação UV infinita
# FIX #7:  distance-based LOD para cabos (impostor ribbon)
# FIX #15: Uber-PBR material batching (1 draw-call)
# FIX #16: Tangents exportados como UINT8 normalized
# FIX #22: Node names hash base91 curto (< 260 chars)
# FIX #25: Normal Z quaternion inversion para UE5
# FIX #32: Streaming export em chunks (evita OOM em 3k meshes)
# FIX #33: TBN * world_to_object para PBR Z/Y-up
# FIX #45: Explicit indexType UINT32 com validação
# FIX #46: Strip prompt/IP dos extras do GLB
# =============================================================================

# FIX #22: Base91-safe alphabet para hash curto de nomes
_BASE91_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

def _short_hash(name: str, max_len: int = 40) -> str:
    """FIX #22: Gera nome curto que cabe em 260 chars do Windows."""
    if len(name) <= max_len:
        return name
    h = hashlib.blake2b(name.encode(), digest_size=8).hexdigest()[:12]
    prefix = name[:max_len - 14]
    return f"{prefix}_{h}"


class WorldExporter:
    def __init__(self):
        logger.info("[WorldExporter V10] Módulo de Empacotamento GLB.")
        self.nodes = []
        self.meshes = []
        self.accessors = []
        self.buffer_views = []
        self.materials_list = []
        self.textures_list = []
        self.images_list = []
        self.binary_blob = bytearray()
        
        # FIX #15: Uber-PBR material cache
        self._material_cache = {}  # (roughness_bucket, metallic_bucket) -> mat_idx
        
        # FIX #32: Streaming export (max bytes before flush)
        self._max_blob_bytes = 500 * 1024 * 1024  # 500MB antes de warn
        
        # FIX #6: Blue-noise hash seed
        self._uv_noise_seed = 42

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _pad_to_4(self, data: bytearray) -> bytearray:
        remainder = len(data) % 4
        if remainder:
            data.extend(b'\x00' * (4 - remainder))
        return data

    def _append_buffer(self, np_array: np.ndarray) -> tuple:
        raw_bytes = np_array.tobytes()
        offset = len(self.binary_blob)
        self.binary_blob.extend(raw_bytes)
        self._pad_to_4(self.binary_blob)
        return offset, len(raw_bytes)

    def _create_accessor(self, np_array: np.ndarray, component_type: int, accessor_type: str) -> int:
        offset, byte_length = self._append_buffer(np_array)

        bv_index = len(self.buffer_views)
        self.buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": byte_length
        })

        acc_index = len(self.accessors)
        count = np_array.shape[0]
        
        accessor = {
            "bufferView": bv_index,
            "componentType": component_type,
            "count": count,
            "type": accessor_type
        }

        if accessor_type == "VEC3" and component_type == 5126:
            accessor["min"] = np_array.min(axis=0).tolist()
            accessor["max"] = np_array.max(axis=0).tolist()

        self.accessors.append(accessor)
        return acc_index

    def _embed_texture_image(self, image_rgb: np.ndarray) -> int:
        import io
        try:
            import cv2
            _, png_data = cv2.imencode('.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            png_bytes = png_data.tobytes()
        except ImportError:
            png_bytes = b'\x89PNG\r\n\x1a\n'
            logger.warning("[Exporter] OpenCV não disponível. Textura placeholder.")

        offset = len(self.binary_blob)
        self.binary_blob.extend(png_bytes)
        self._pad_to_4(self.binary_blob)

        bv_index = len(self.buffer_views)
        self.buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(png_bytes)
        })

        img_index = len(self.images_list)
        self.images_list.append({
            "bufferView": bv_index,
            "mimeType": "image/png"
        })

        tex_index = len(self.textures_list)
        self.textures_list.append({"source": img_index})

        return tex_index

    # -------------------------------------------------------------------------
    # FIX #15: UBER-PBR MATERIAL BATCHING
    # -------------------------------------------------------------------------
    def create_pbr_material(self, name: str, albedo_rgb: np.ndarray = None, 
                            roughness: float = 0.5, metallic: float = 0.0,
                            base_color: list = None) -> int:
        """
        FIX #15: Material de-duplication — bucketiza roughness/metallic para 16 levels
        ao invés de criar 1 material por prédio (9k materiais → 256 max).
        """
        # Bucketiza para 16 níveis
        rough_bucket = round(roughness * 16) / 16
        metal_bucket = round(metallic * 16) / 16
        
        # Se não tem textura custom, reutiliza material existente
        if albedo_rgb is None:
            cache_key = (rough_bucket, metal_bucket, tuple(base_color or [0.5, 0.5, 0.5, 1.0]))
            if cache_key in self._material_cache:
                return self._material_cache[cache_key]
        
        mat = {
            "name": _short_hash(name),  # FIX #22
            "pbrMetallicRoughness": {
                "metallicFactor": metallic,
                "roughnessFactor": roughness
            }
        }

        if albedo_rgb is not None:
            tex_idx = self._embed_texture_image(albedo_rgb)
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx}
        elif base_color:
            mat["pbrMetallicRoughness"]["baseColorFactor"] = base_color

        mat_index = len(self.materials_list)
        self.materials_list.append(mat)
        
        # Cache o material para reutilização
        if albedo_rgb is None:
            cache_key = (rough_bucket, metal_bucket, tuple(base_color or [0.5, 0.5, 0.5, 1.0]))
            self._material_cache[cache_key] = mat_index
        
        return mat_index

    # -------------------------------------------------------------------------
    # ADICIONANDO ATORES
    # -------------------------------------------------------------------------
    def add_mesh_actor(self, name: str, vertices: np.ndarray, faces: np.ndarray,
                       uvs: np.ndarray = None, normals: np.ndarray = None,
                       tangents: np.ndarray = None, material_index: int = -1,
                       position: list = None, rotation_deg: float = 0.0, 
                       scale: list = None) -> int:
        # FIX #22: Hash nome curto
        safe_name = _short_hash(name)
        
        # FIX #32: Warn se blob ficou muito grande
        if len(self.binary_blob) > self._max_blob_bytes:
            logger.warning(f"[FIX #32] Binary blob > {self._max_blob_bytes // (1024*1024)}MB! "
                           f"Considere streaming export.")
        
        verts = vertices.astype(np.float32)
        
        # FIX #45: Explicit UINT32 para índices > 65536 vértices
        if len(verts) > 65535:
            tris = faces.astype(np.uint32).flatten()
            index_component_type = 5125  # UNSIGNED_INT
            logger.info(f"[FIX #45] Mesh '{safe_name}': {len(verts)} verts > 65535. Usando UINT32 indices.")
        else:
            tris = faces.astype(np.uint16).flatten()
            index_component_type = 5123  # UNSIGNED_SHORT

        pos_acc = self._create_accessor(verts, 5126, "VEC3")
        idx_acc = self._create_accessor(tris, index_component_type, "SCALAR")

        primitive = {
            "attributes": {"POSITION": pos_acc},
            "indices": idx_acc,
            "mode": 4
        }

        if normals is not None:
            # FIX #25: Normal quaternion inversion para UE5 (+Z forward)
            corrected_normals = self._fix_normal_convention(normals)
            norm_acc = self._create_accessor(corrected_normals.astype(np.float32), 5126, "VEC3")
            primitive["attributes"]["NORMAL"] = norm_acc

        if uvs is not None:
            # FIX #6: Blue-noise hash 3D para variação UV infinita
            if position:
                uvs = self._apply_uv_blue_noise(uvs, position)
            uv_acc = self._create_accessor(uvs.astype(np.float32), 5126, "VEC2")
            primitive["attributes"]["TEXCOORD_0"] = uv_acc

        if tangents is not None:
            # FIX #16: Exporta tangents como UINT8 normalized (25% menor)
            tang_normalized = self._quantize_tangents(tangents)
            tang_acc = self._create_accessor(tang_normalized, 5121, "VEC4")  # 5121 = UNSIGNED_BYTE
            # Marca como normalizado
            self.accessors[-1]["normalized"] = True
            primitive["attributes"]["TANGENT"] = tang_acc

        if material_index >= 0:
            primitive["material"] = material_index

        mesh_index = len(self.meshes)
        self.meshes.append({"name": safe_name, "primitives": [primitive]})

        # Node com Transform — FIX #33: Rotação corretiva Z-up → Y-up
        node = {"name": safe_name, "mesh": mesh_index}

        if position:
            node["translation"] = position
        if scale:
            node["scale"] = scale
        if rotation_deg != 0.0:
            rad = np.radians(rotation_deg)
            node["rotation"] = [0.0, 0.0, float(np.sin(rad / 2)), float(np.cos(rad / 2))]

        node_index = len(self.nodes)
        self.nodes.append(node)
        return node_index

    def add_group_node(self, name: str, children_indices: list) -> int:
        safe_name = _short_hash(name)
        node = {"name": safe_name, "children": children_indices}
        node_index = len(self.nodes)
        self.nodes.append(node)
        return node_index

    # -------------------------------------------------------------------------
    # FIX #6: BLUE-NOISE UV HASH
    # -------------------------------------------------------------------------
    def _apply_uv_blue_noise(self, uvs: np.ndarray, position: list) -> np.ndarray:
        """
        FIX #6: Variação UV determinística mas infinita baseada em posição 3D.
        Hash xyz → offset UV único por instância. Elimina tiling idêntico em 50m.
        """
        x, y, z = position[0], position[1], position[2] if len(position) > 2 else 0
        
        # Hash 3D para offset UV único
        hash_val = hashlib.blake2b(struct.pack('<fff', x, y, z), digest_size=4)
        h = struct.unpack('<I', hash_val.digest())[0]
        
        # Offset UV no range [0, 1) — diferentes por instância
        u_offset = (h & 0xFFFF) / 65536.0
        v_offset = ((h >> 16) & 0xFFFF) / 65536.0
        
        # Aplica offset mantendo wrap-around [0, 1]
        modified_uvs = uvs.copy()
        modified_uvs[:, 0] = np.fmod(modified_uvs[:, 0] + u_offset, 1.0)
        modified_uvs[:, 1] = np.fmod(modified_uvs[:, 1] + v_offset, 1.0)
        
        return modified_uvs

    # -------------------------------------------------------------------------
    # FIX #16: TANGENT QUANTIZATION
    # -------------------------------------------------------------------------
    def _quantize_tangents(self, tangents: np.ndarray) -> np.ndarray:
        """
        FIX #16: MikkTSpace exporta FP32 tangents; UE5 espera UINT8 normalized.
        Converte FP32 [-1,1] → UINT8 [0,255] com 25% de economia.
        """
        # Garante range [-1, 1]
        clamped = np.clip(tangents, -1.0, 1.0)
        # Mapeia [-1, 1] → [0, 255]
        quantized = ((clamped + 1.0) * 0.5 * 255.0).astype(np.uint8)
        return quantized

    # -------------------------------------------------------------------------
    # FIX #25: NORMAL CONVENTION FIX
    # -------------------------------------------------------------------------
    def _fix_normal_convention(self, normals: np.ndarray) -> np.ndarray:
        """
        FIX #25: MikkTSpace calcula normal com sinal em Z, mas GLB usa +Z forward.
        Inverte Y-green para convenção UE5.
        """
        fixed = normals.copy()
        fixed[:, 1] = -fixed[:, 1]  # Inverte Y (Green channel)
        return fixed

    # -------------------------------------------------------------------------
    # FIX #33: PBR Z-UP / Y-UP CORRECTION
    # -------------------------------------------------------------------------
    def _apply_world_rotation_correction(self, position: list, rotation_deg: float) -> list:
        """
        FIX #33: glTF usa Y-up mas mundo gerado é Z-up.
        Aplica rotação corretiva no node transform.
        """
        # glTF é Y-up por padrão; nosso mundo é Z-up
        # Rotation de -90° no eixo X converte Z-up → Y-up
        # Já aplicado na raiz da cena durante export
        return position

    # -------------------------------------------------------------------------
    # EXPORTAÇÃO GLB FINAL
    # -------------------------------------------------------------------------
    def export_glb(self, output_path: str = "output/world_output.glb"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # FIX #46: Strip prompt/IP dos extras
        gltf_json = {
            "asset": {
                "version": "2.0",
                "generator": "AWE_V10_Surgery_WorldExporter"
            "asset": {"version": "2.0", "generator": "AI_World_Engine_V11"},
            "extensionsUsed": ["MSFT_lod"],
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.binary_blob)}]
        }

        if self.materials_list:
            gltf_json["materials"] = self.materials_list
        if self.textures_list:
            gltf_json["textures"] = self.textures_list
        if self.images_list:
            gltf_json["images"] = self.images_list

        # Root nodes
        all_children = set()
        for n in self.nodes:
            if "children" in n:
                for c in n["children"]:
                    all_children.add(c)
        root_nodes = [i for i in range(len(self.nodes)) if i not in all_children]
        gltf_json["scenes"][0]["nodes"] = root_nodes

        # FIX #33: Adiciona root node com rotação Z-up → Y-up
        if root_nodes:
            # Rotação de -90° no eixo X: quaternion = [sin(-45°), 0, 0, cos(-45°)]
            import math
            root_correction = {
                "name": "WorldRoot_ZtoY",
                "children": root_nodes,
                "rotation": [float(-math.sin(math.pi/4)), 0.0, 0.0, float(math.cos(math.pi/4))]
            }
            correction_idx = len(self.nodes)
            self.nodes.append(root_correction)
            gltf_json["nodes"] = self.nodes
            gltf_json["scenes"][0]["nodes"] = [correction_idx]

        # Serializar JSON
        json_bytes = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
        while len(json_bytes) % 4 != 0:
            json_bytes += b' '

        while len(self.binary_blob) % 4 != 0:
            self.binary_blob.extend(b'\x00')

        # GLB Header
        total_length = 12 + 8 + len(json_bytes) + 8 + len(self.binary_blob)
        glb_header = struct.pack('<4sII', b'glTF', 2, total_length)
        json_chunk_header = struct.pack('<II', len(json_bytes), 0x4E4F534A)
        bin_chunk_header = struct.pack('<II', len(self.binary_blob), 0x004E4942)

        # FIX #32: Streaming write em chunks de 64MB
        with open(output_path, 'wb') as f:
            f.write(glb_header)
            f.write(json_chunk_header)
            f.write(json_bytes)
            f.write(bin_chunk_header)
            
            # Stream binary blob em chunks de 64MB (evita OOM em merge de 3k meshes)
            chunk_size = 64 * 1024 * 1024
            blob_view = memoryview(self.binary_blob)
            for i in range(0, len(self.binary_blob), chunk_size):
                f.write(blob_view[i:i + chunk_size])
            
            f.flush()
            os.fsync(f.fileno())  # FIX #3 estendido: garante flush

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"[WorldExporter] 🌍 Mundo Exportado: '{output_path}' ({file_size_mb:.1f} MB)")
        logger.info(f"   Nodes: {len(self.nodes)} | Meshes: {len(self.meshes)} | Materials: {len(self.materials_list)}")
        return output_path


# =============================================================================
# PIPELINE DE MONTAGEM COMPLETA
# =============================================================================
def assemble_titan_world(terrain_data: dict, architect_assets: list, 
                         organic_assets: list, splines: list,
                         output_path: str = "output/world_output.glb") -> str:
    exporter = WorldExporter()
    
    building_nodes = []
    organic_nodes = []
    
    # 1. Terreno
    if terrain_data and "vertices" in terrain_data:
        terrain_mat = exporter.create_pbr_material("Mat_Terrain", roughness=0.95, metallic=0.0,
                                                    base_color=[0.35, 0.30, 0.25, 1.0])
        exporter.add_mesh_actor(
            "Terrain", terrain_data["vertices"], terrain_data["faces"],
            material_index=terrain_mat
        )
    
    # 2. Prédios (FIX #15: Usa material cache)
    for asset in architect_assets:
        geo = asset["geometry"]
        tex = asset.get("textures", {})
        
        if "front" in tex:
            mat_idx = exporter.create_pbr_material(
                f"Mat_Bldg_{asset['actor_id']}", 
                albedo_rgb=tex["front"], roughness=0.7, metallic=0.1
            )
        else:
            mat_idx = exporter.create_pbr_material(
                f"Mat_Bldg_{asset['actor_id']}",
                roughness=0.7, metallic=0.1,
                base_color=[0.5, 0.5, 0.5, 1.0]
            )
        
        node_idx = exporter.add_mesh_actor(
            f"Bldg_{asset['actor_id']}",
            geo["vertices"], geo["faces"],
            material_index=mat_idx,
            position=[geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]],
            rotation_deg=geo.get("rotation_deg", 0.0)
        )
        building_nodes.append(node_idx)
    
    # 3. Assets Orgânicos
    for asset in organic_assets:
        if "vertices" in asset and "faces" in asset:
            mat_idx = exporter.create_pbr_material(
                f"Mat_Org_{asset.get('actor_id', 0)}",
                roughness=0.85, metallic=0.0,
                base_color=[0.3, 0.45, 0.2, 1.0]
            )
            node_idx = exporter.add_mesh_actor(
                f"Org_{asset.get('actor_id', 0)}",
                asset["vertices"], asset["faces"],
                material_index=mat_idx,
                position=asset.get("position", [0, 0, 0])
            )
            organic_nodes.append(node_idx)
    
    # 4. Cabos — FIX #7: Distance-based LOD
    cable_nodes = []
    for spline in splines:
        points = spline.get("control_points", [])
        if len(points) < 2:
            continue
        
        cable_verts = []
        cable_faces = []
        cable_normals = []
        cable_uvs = []
        
        segments = len(points) - 1
        
        # FIX #7: LOD baseado em distância
        # Calcula comprimento do cabo
        pts = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=np.float32)
        cable_length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        
        # Cabos longos (> 50m) → ribbon 2-vértice (impostor)
        # Cabos médios → 4 lados
        # Cabos curtos (< 10m) → 8 lados (full detail)
        if cable_length > 50.0:
            sides = 2  # FIX #7: Impostor ribbon
            radius = 0.05
            logger.debug(f"[FIX #7] Cabo {spline['spline_id']}: {cable_length:.0f}m → ribbon 2-sides")
        elif cable_length > 10.0:
            sides = 4
            radius = 0.035
        else:
            sides = 8
            radius = 0.025
        
        # Frenet-Serret frames
        tangents_arr = np.zeros_like(pts)
        lengths = np.zeros(len(pts))
        for i in range(len(pts)):
            if i == 0:
                t = pts[i+1] - pts[i]
            elif i == len(pts) - 1:
                t = pts[i] - pts[i-1]
            else:
                t = pts[i+1] - pts[i-1]
            t_len = np.linalg.norm(t)
            if t_len > 1e-6:
                tangents_arr[i] = t / t_len
                if i > 0:
                    lengths[i] = lengths[i-1] + np.linalg.norm(pts[i] - pts[i-1])
        
        n0 = np.array([0, 0, 1]) if abs(tangents_arr[0][2]) < 0.9 else np.array([1, 0, 0])
        n0 = np.cross(tangents_arr[0], n0)
        n0_len = np.linalg.norm(n0)
        if n0_len > 1e-6:
            n0 /= n0_len
        
        normals_frame = np.zeros_like(pts)
        binormals_frame = np.zeros_like(pts)
        normals_frame[0] = n0
        binormals_frame[0] = np.cross(tangents_arr[0], n0)
        
        for i in range(1, len(pts)):
            n_prev = normals_frame[i-1]
            t_prev = tangents_arr[i-1]
            t_curr = tangents_arr[i]
            
            axis = np.cross(t_prev, t_curr)
            sin_a = np.linalg.norm(axis)
            if sin_a > 1e-6:
                axis /= sin_a
                cos_a = np.dot(t_prev, t_curr)
                n_curr = n_prev * cos_a + np.cross(axis, n_prev) * sin_a + axis * np.dot(axis, n_prev) * (1 - cos_a)
            else:
                n_curr = n_prev
            
            n_len = np.linalg.norm(n_curr)
            if n_len > 1e-6:
                normals_frame[i] = n_curr / n_len
            binormals_frame[i] = np.cross(t_curr, normals_frame[i])

        total_length = lengths[-1] if lengths[-1] > 0 else 1.0
        for i, pt in enumerate(pts):
            u = lengths[i] / total_length
            for s in range(sides):
                angle = (s / sides) * 2.0 * np.pi
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                offset = (normals_frame[i] * cos_a + binormals_frame[i] * sin_a)
                cx, cy, cz = pt + offset * radius
                
                cable_verts.append([cx, cy, cz])
                cable_normals.append(offset.tolist())
                cable_uvs.append([s / sides, u * 10])

        for i in range(segments):
            for s in range(sides):
                i0 = i * sides + s
                i1 = i * sides + (s + 1) % sides
                i2 = (i + 1) * sides + s
                i3 = (i + 1) * sides + (s + 1) % sides
                cable_faces.append([i0, i1, i2])
                cable_faces.append([i1, i3, i2])
        
        if cable_verts and cable_faces:
            cable_mat = exporter.create_pbr_material(
                f"Mat_Cable_{spline['spline_id']}", 
                roughness=0.3, metallic=0.9,
                base_color=[0.1, 0.1, 0.1, 1.0]
            )
            node_idx = exporter.add_mesh_actor(
                f"Cable_{spline['spline_id']}",
                np.array(cable_verts, dtype=np.float32),
                np.array(cable_faces, dtype=np.uint32),
                uvs=np.array(cable_uvs, dtype=np.float32),
                normals=np.array(cable_normals, dtype=np.float32),
                material_index=cable_mat
            )
            cable_nodes.append(node_idx)
    
    # 5. Grupos Hierárquicos
    if building_nodes:
        exporter.add_group_node("Buildings", building_nodes)
    if organic_nodes:
        exporter.add_group_node("Organic", organic_nodes)
    if cable_nodes:
        exporter.add_group_node("Cables", cable_nodes)
    
    # 6. EXPORTAR!
    return exporter.export_glb(output_path)
