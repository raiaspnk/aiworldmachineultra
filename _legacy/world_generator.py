import os
import cv2
import time
import trimesh
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from awe_logger import AWELogger

# Logger de módulo (compartilhado por todas as funções deste arquivo)
_log = AWELogger("WORLD")

def optimize_topology(verts, faces, target_reduction=0.7):
    """
    [ARTISANAL TOPOLOGY] Aplica Quadric Edge Collapse Decimation via Open3D.
    Reduz a contagem de polígonos brutalmente (em áreas retas/planas)
    mas preserva a alta densidade nas quinas e nos detalhes do Marigold.
    """
    if len(faces) == 0:
        return verts, faces
    try:
        import open3d as o3d
        # Fast path if no reduction is needed or very few faces
        if len(faces) < 100:
            return verts, faces
            
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        target_triangles = max(4, int(len(faces) * (1.0 - target_reduction)))
        
        decimated = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles,
            maximum_error=0.005,
            boundary_weight=1.0
        )
        
        return np.asarray(decimated.vertices), np.asarray(decimated.triangles)
    except ImportError:
        return verts, faces

def recalculate_uvs(verts, offset_x, offset_y, scale):
    """
    Recalcula as coordenadas UV (Top-Down) puramente baseadas na 
    posição Mundial (World Space) dos novos vértices decimação.
    """
    if len(verts) == 0:
        return np.array([])
    x = verts[:, 0]
    y = verts[:, 1]
    u = (x - offset_x) / scale + 0.5
    v = 0.5 - (y - offset_y) / scale
    return np.column_stack((u, v))

def extract_semantic_prefabs(scene: trimesh.Scene, sam3_metadata: Optional[Dict[str, Any]], output_dir: str, scale: float = 1.0):
    """
    [PHASE V5: WORLD ASSEMBLER SPAWNER]
    Lê os objetos perfeitamente processados pelo Hunyuan3D-2 no metadado do SAM 3 e 
    os posiciona corretamente no dicionário de spawn (world_data.json) para que a
    game engine os instancie. Não corta mais a cena raiz 2.5D.
    """
    import json
    
    world_data = {
        "engine_version": "AI_World_Engine_V5",
        "objects": []
    }
    
    _log.phase("V5_FAB").info(f"Iniciando registro de spawns no Spawner Registry...")
    
    # 1. Adicionar o terreno principal (chão)
    if scene and len(scene.geometry) > 0:
        world_data["objects"].append({
            "type": "terrain",
            "id": "main_terrain",
            "filename": "world_mesh.glb", # O arquivo principal de terreno exportado
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "random_uv_offset": {"u": 0.0, "v": 0.0} # Terreno principal não usa random_uv_offset
        })
        
    has_sam_masks = (
        sam3_metadata 
        and "objects" in sam3_metadata 
        and len(sam3_metadata["objects"]) > 0
    )
    
    if has_sam_masks:
        sam_objs = sam3_metadata["objects"]
        _log.phase("V5_FAB").ok(f"Registrando {len(sam_objs)} objetos sólidos do Hunyuan3D-2...")
        
        for obj in sam_objs:
            if "glb_path" not in obj:
                # Objeto foi podado pelo LOD (muito pequeno) ou falhou no H3D2
                continue
                
            x, y, w, h = obj["bbox"]
            cx = x + w / 2.0
            cy = y + h / 2.0
            
            concept = obj.get("concept_safe", "object")
            obj_id = obj["id"]
            
            # A coordenada X, Y no jogo é baseada na escala do pixels.
            # No terrain, 1 chunk (eff_res px) = scale unidades.
            # Por simplicidade, assumimos chunk_resolution padrão de 1024.
            eff_res = 1023.0 
            world_x = cx * (scale / eff_res)
            world_y = cy * (scale / eff_res)
            
            # O terreno foi achatado para o nivel da rua (Z ou Y) via inpaint.
            world_z = 0.0 
            
            # [V8 Enterprise]: Instance UV Randomization
            # Usando coordenadas espaciais para gerar um float pseudo-random [0.0 - 1.0] persistente.
            # O Material Shader no UE5 usará isso via PerInstanceRandom ou Node Parameter.
            uv_offset_u = float(max(0.0, min(abs(np.sin(world_x * 12.9898 + world_y * 78.233)), 1.0)))
            uv_offset_v = float(max(0.0, min(abs(np.cos(world_x * 12.9898 + world_y * 78.233)), 1.0)))
            
            world_data["objects"].append({
                "type": concept,
                "id": obj_id,
                "filename": obj["glb_path"],
                "position": {"x": world_x, "y": world_z, "z": world_y},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                "random_uv_offset": {"u": uv_offset_u, "v": uv_offset_v}
            })
    else:
        # Fallback caso não use SAM - exportar instâncias burras do V4 Grouping
        _log.phase("FAB").warn("Sem objetos SAM 3 — instanciando cena crua...")
        # ... (Mantendo export base do terreno)
    
    json_path = os.path.join(output_dir, "world_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(world_data, f, indent=4)
        
    _log.phase("V5_FAB").ok(f"Manifesto JSON V5 exportado com {len(world_data['objects'])} Spawns.")
    return json_path

def generate_landscape_mesh(
    image_path: str,
    depth_map_path: str,
    output_path: str,
    marigold_map_path: Optional[str] = None,
    max_height: float = 0.5,
    chunk_resolution: int = 1024,
    smoothing_iterations: int = 3,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale: float = 1.0,
    sam3_metadata: Optional[Dict[str, Any]] = None,
    sam3_masks_path: Optional[str] = None
) -> bool:
    """
    Gera uma malha 3D (.glb) de um cenário completo aplicando "Depth Displacement".
    Pega uma grade 2D plana, e empurra os vértices no eixo Z baseado no tom de cinza
    do mapa de profundidade, aplicando a textura colorida original por cima.
    
    Args:
        image_path: Caminho da imagem RGB original (textura).
        depth_map_path: Caminho do mapa de profundidade (grayscale).
        output_path: Caminho final do arquivo .glb.
        max_height: Altura máxima de extrusão (baseado no depth 255).
        mesh_resolution: Quantidade X e Y de polígonos da malha (ex: 512x512).
        smoothing_iterations: Iterações de Laplacian Smoothing para não ficar pontiagudo.
        offset_x: Deslocamento no eixo X para Seamless Tiling de chunks adjacentes.
        offset_y: Deslocamento no eixo Y para Seamless Tiling de chunks adjacentes.
        scale: Escala física do World space (1.0 = bloco de 1x1 unidade).
    """
    _log.phase("INIT").info(f"Extrusão de terreno iniciada ({chunk_resolution}x{chunk_resolution}px)")
    
    try:
        if not os.path.exists(depth_map_path):
            _log.phase("INIT").error(f"Mapa de profundidade não encontrado: {depth_map_path}")
            return False
            
        # 1. Carregar a textura e o depth map
        img_color = Image.open(image_path).convert("RGB")
        
        # [ANTI-STAIRCASE] Ler NPZ bruto métrico (float) ou Imagem 16-bits para DA3
        if depth_map_path.endswith('.npz'):
            depth_data = np.load(depth_map_path)
            img_depth = depth_data['depth']
            
            # INVERTER PROFUNDIDADE MÉTRICA: Elementos próximos (valores menores)
            # devem se tornar as extrusões mais altas do terreno (valores maiores).
            min_d, max_d = img_depth.min(), img_depth.max()
            if max_d > min_d:
                img_depth = (max_d - img_depth) / (max_d - min_d)
            else:
                img_depth = np.zeros_like(img_depth)
        else:
            img_depth = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
            
        if img_depth is None:
            _log.phase("INIT").error(f"Falha ao ler mapa de profundidade: {depth_map_path}")
            return False
            
        # Padronizar como float32 para as operações de matriz
        img_depth = img_depth.astype(np.float32)

        # ─── MÓDULO MARIGOLD (Fusão Macro/Micro Depth) ───
        if marigold_map_path and os.path.exists(marigold_map_path):
            _log.phase("MARIGOLD").info("Fundindo Macro-Depth (DA3) com Micro-Displacement (Marigold)...")
            try:
                if marigold_map_path.endswith('.npz'):
                    mari_data = np.load(marigold_map_path)
                    img_mari = mari_data['depth'].astype(np.float32)
                else:
                    img_mari = cv2.imread(marigold_map_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                
                # Normaliza o Marigold entre 0 e 1 (Marigold Affine Invariant não precisa inverte, o claro já = perto)
                min_m, max_m = img_mari.min(), img_mari.max()
                if max_m > min_m:
                    img_mari = (img_mari - min_m) / (max_m - min_m)
                    
                # Redimensiona o Marigold para ter EXATAMENTE o mesmo tamanho do DA3
                img_mari_resized = cv2.resize(img_mari, (img_depth.shape[1], img_depth.shape[0]), interpolation=cv2.INTER_LANCZOS4)

                # A MÁGICA DA FUSÃO (80% Estrutura DA3, 20% Detalhe Fino Marigold)
                # O DA3 faz o prédio existir, o Marigold esculpe os tijolos
                img_depth = (img_depth * 0.8) + (img_mari_resized * 0.2)
                print("   ✅ Fusão de Tensores Geométricos Concluída.")

            except Exception as e:
                _log.phase("MARIGOLD").warn(f"Falha ao fundir Marigold, usando apenas DA3: {e}")
        # ────────────────────────────────────────────────────────
        
        # [ANTI-MELTING] Aplicar filtro High-Pass (Sharpen) na profundidade
        # Isso acentua as quinas dos prédios e "cliva" transições suaves entre parede e chão
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_depth_hpass = cv2.filter2D(img_depth, -1, kernel_sharpen)
        
        # ─── V5 ARCHITECT TITAN EDITION: GROUND SOCKETING ───
        # Calcula a média (inpaint) da fundação e cria um PLATÔ com falloff nas bordas
        if sam3_metadata and sam3_masks_path and os.path.exists(sam3_masks_path):
            try:
                masks_data = np.load(sam3_masks_path)
                combined_mask = np.zeros_like(img_depth_hpass, dtype=np.uint8)
                
                for k in masks_data.files:
                    mask_layer = masks_data[k]
                    if mask_layer.shape[:2] != combined_mask.shape[:2]:
                        mask_layer = cv2.resize(mask_layer.astype(np.uint8), (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_layer = mask_layer.astype(np.uint8)
                        
                    # Dilatar a máscara em leve proporção para expandir o "socket"
                    kernel_size = max(3, int(img_depth_hpass.shape[1] * 0.015) | 1)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_dilated = cv2.dilate(mask_layer, kernel, iterations=1)
                    combined_mask[mask_dilated > 0] = 255
                    
                # Aplicar achatamento base (Inpaint Téléa puxa do nível da rua para dentro)
                img_inpaint = cv2.inpaint(img_depth_hpass, combined_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
                
                # Falloff (Suavização das bordas para o chão "abraçar" a fundação)
                blur_kernel = max(11, int(img_depth_hpass.shape[1] * 0.03) | 1)
                img_blurred = cv2.GaussianBlur(img_inpaint, (blur_kernel, blur_kernel), 0)
                
                # Alpha mask para mesclar fundação com o resto (bordas feathered)
                alpha = cv2.GaussianBlur(combined_mask.astype(np.float32), (blur_kernel, blur_kernel), 0) / 255.0
                
                # Blend do terreno original intocado com os Socket Platos suavizados
                img_depth_hpass = img_depth_hpass * (1.0 - alpha) + img_blurred * alpha
                
                _log.phase("V5_TITAN").ok("Ground Socketing aplicado com sucesso! (Platôs com Falloff gerados).")
            except Exception as e:
                _log.phase("V5_TITAN").warn(f"Falha ao realizar Ground Socketing: {e}")
                
        # Extrair dimensões base da imagem RGB
        orig_w, orig_h = img_color.size
        
        # Para garantir que os tiles não tenham UMA fresta de divisão, precisamos que o último 
        # vértice do Tile 0 seja IDÊNTICO na posição X,Y,Z ao primeiro vértice do Tile 1.
        # Portanto, há um Overlap obrigatório de 1 pixel em toda as junções.
        overlap = 1
        eff_res = chunk_resolution - overlap
        
        # Validar tamanho da imagem vs chunk_resolution
        chunks_x = max(1, orig_w // eff_res)
        chunks_y = max(1, orig_h // eff_res)
        
        # Redimensionar para o novo tamanho costurado
        new_w = chunks_x * eff_res + overlap
        new_h = chunks_y * eff_res + overlap
        
        img_color = img_color.resize((new_w, new_h), Image.LANCZOS)
        img_depth_hpass = cv2.resize(img_depth_hpass, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        _log.phase("INIT").info(f"Seamless Auto-Tiling: {chunks_x}×{chunks_y} grid — {chunks_x * chunks_y} chunks de {chunk_resolution}px (1px overlap)")
        
        # Iniciar container da Scene (para mesclar N mundos em 1 único arquivo exportado)
        scene = trimesh.Scene()
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Setup CUDA
        import torch
        import torch.nn.functional as F
        import sys
        try:
            from monster_core import generate_world_geometry_pipeline, gpu_ransac_hard_surface
            use_cuda = torch.cuda.is_available()
        except ImportError as e:
            import traceback
            _log.phase("INIT").error(f"MonsterCore V3 não encontrado ou erro de importação. Erro real: {e}")
            _log.phase("INIT").error(f"Traceback: {traceback.format_exc()}")
            use_cuda = False
            
        if not use_cuda:
            _log.phase("INIT").error("CUDA/MonsterCore indisponível. Abortando geração de mundo massivo.")
            return False
            
        # Loop de Tiling (Gerar Fatias)
        for cy in range(chunks_y):
            for cx in range(chunks_x):
                _chunk_t0 = time.perf_counter()
                _log.phase(f"Chunk {cx},{cy}").separator(title=f"CHUNK [{cx},{cy}] de [{chunks_x-1},{chunks_y-1}]")
                
                # Coordenadas Crop com Overlap Matemático (evita rasgar a malha no horizonte)
                x0 = cx * eff_res
                y0 = cy * eff_res
                x1 = x0 + chunk_resolution
                y1 = y0 + chunk_resolution
                
                # 1.1 Fatiar Textura e Profundidade
                crop_color = img_color.crop((x0, y0, x1, y1))
                crop_depth = img_depth_hpass[y0:y1, x0:x1]
                
                # Normalização adaptativa por chunk
                max_val = float(np.max(crop_depth))
                if max_val == 0.0: max_val = 1.0
                depth_norm = crop_depth.astype(np.float32) / max_val
                depth_tensor = torch.from_numpy(depth_norm).cuda()
                
                # --- [MONSTER CORE V4: Semantic Geometry Decoupling] ---
                import torch.nn.functional as F
                kernel_sz = 31
                pad = kernel_sz // 2
                depth_unsqueezed = depth_tensor.unsqueeze(0).unsqueeze(0)
                eroded = -F.max_pool2d(-depth_unsqueezed, kernel_sz, stride=1, padding=pad)
                terrain_depth_tensor = F.max_pool2d(eroded, kernel_sz, stride=1, padding=pad).squeeze()
                
                # Mascara Booleana: Prédios/Estruturas (Altura dif > 2% do chão)
                # Adaptado: Menos estrito se a cena foi muito comprimida no norm
                building_threshold = 0.02
                building_mask_tensor = (depth_tensor - terrain_depth_tensor) > building_threshold
                
                overlap_compensation = float(eff_res) / float(chunk_resolution - 1)
                offset_x_world = float(cx) * overlap_compensation * scale
                offset_y_world = float(chunks_y - 1 - cy) * overlap_compensation * scale # Inverte eixo Y
                
                # Ajustar max_height pela escala para manter a proporção (se scale=100, altura máxima=50)
                actual_max_height = max_height * scale
                
                # 2.A: Forjar Geometria do Chão Limpo (Terrain Mesh)
                t_verts_pt, t_faces_pt, t_norms_pt, t_foliage_pt, _ = generate_world_geometry_pipeline(
                    depth_map=terrain_depth_tensor, 
                    max_height=actual_max_height, offset_x=offset_x_world, offset_y=offset_y_world,
                    scale=scale, smooth_iters=smoothing_iterations, smooth_lambda=0.5
                )
                
                # 2.B: Forjar Geometria da Cidade (Prédios e Muros Triplanares)
                b_verts_pt, b_faces_pt, b_norms_pt, b_foliage_pt, b_mat_pt = generate_world_geometry_pipeline(
                    depth_map=depth_tensor, 
                    max_height=actual_max_height, offset_x=offset_x_world, offset_y=offset_y_world,
                    scale=scale, smooth_iters=0, smooth_lambda=0.5 # Muros retos, não suavizar prédios
                )

                # --- [ARTISANAL HARD-SURFACE: GPU RANSAC] ---
                # Cura o "Queijo Derretido" das paredes esticadas. Força os vértices que estão
                # na vertical (salto do muro) a se alinharem num plano matemático reto e brutalista,
                # mantendo os detalhes finos onde a máscara de ruído do Marigold atuou.
                with _log.phase(f"Chunk {cx},{cy}").timer("RANSAC"):
                    try:
                        ransac_dist = 0.02 * scale
                        b_verts_pt = gpu_ransac_hard_surface(
                            vertices=b_verts_pt,
                            distance_threshold=ransac_dist,
                            num_iterations=1200,
                            batch_size=200
                        )
                        _log.phase(f"Chunk {cx},{cy}").ok("Paredes retificadas na GPU (RANSAC OK)")
                    except Exception as e:
                        _log.phase(f"Chunk {cx},{cy}").warn(f"RANSAC falhou, usando geometria bruta: {e}")
                # --------------------------------------------
                _chunk_gpu_ms = (time.perf_counter() - _chunk_t0) * 1000
                _log.phase(f"Chunk {cx},{cy}").metric("GPU Total", f"{_chunk_gpu_ms:.1f}", "ms")
                
                t_verts = t_verts_pt.cpu().numpy()
                t_faces = t_faces_pt.cpu().numpy()
                
                b_verts = b_verts_pt.cpu().numpy()
                b_faces = b_faces_pt.cpu().numpy()
                b_mat = b_mat_pt.cpu().numpy()
                bldg_mask = building_mask_tensor.cpu().numpy().flatten()
                
                # --- Filtro V4: Isolar Prédios e Muros ---
                face_v0 = b_faces[:, 0]
                face_v1 = b_faces[:, 1]
                face_v2 = b_faces[:, 2]
                
                is_roof = bldg_mask[face_v0] | bldg_mask[face_v1] | bldg_mask[face_v2]
                is_wall = (b_mat == 1)
                
                bldg_roof_faces = b_faces[is_roof & ~is_wall]
                bldg_wall_faces = b_faces[is_wall]
                
                # 3. As coordenadas UV são recalculadas em recalculate_uvs() APÓS a decimação.
                # Não criar uvs fixos aqui — eles seriam incompatíveis com vertices decimados.
                normal_path = os.path.join(base_dir, f"{base_name}_normal_{cx}_{cy}.png")
                normals_img = (t_norms_pt.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(normal_path, cv2.cvtColor(normals_img, cv2.COLOR_RGB2BGR))
                img_normal = Image.open(normal_path).convert("RGB")
                
                foliage_path = os.path.join(base_dir, f"{base_name}_foliage_{cx}_{cy}.png")
                foliage_img = (t_foliage_pt.cpu().numpy() * 255).astype(np.uint8)
                kernel_morph = np.ones((5,5), np.uint8)
                foliage_img = cv2.morphologyEx(foliage_img, cv2.MORPH_OPEN, kernel_morph) 
                foliage_img = cv2.morphologyEx(foliage_img, cv2.MORPH_CLOSE, kernel_morph)
                cv2.imwrite(foliage_path, foliage_img)
                
                # Material Base Top-Down
                material_td = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=crop_color,
                    normalTexture=img_normal,
                    roughnessFactor=0.8,
                    metallicFactor=0.1
                )
                
                # Material Triplanar (Para os Muros Cortados da Cidade)
                material_wall = trimesh.visual.material.PBRMaterial(
                    name=f"Triplanar_Wall",
                    baseColorFactor=[100, 100, 100, 255],
                    metallicFactor=0.2,
                    roughnessFactor=0.5
                )

                with _log.phase(f"Chunk {cx},{cy}").timer("Decimação"):
                    t_verts_opt, t_faces_opt = optimize_topology(t_verts, t_faces, target_reduction=0.85)
                t_uvs_opt = recalculate_uvs(t_verts_opt, offset_x_world, offset_y_world, scale)
                _log.phase(f"Chunk {cx},{cy}").metric("Terrain", f"{len(t_faces):,} → {len(t_faces_opt):,}", " faces")
                
                # Construir Meshes
                t_mesh = trimesh.Trimesh(vertices=t_verts_opt, faces=t_faces_opt, process=False)
                t_mesh.visual = trimesh.visual.TextureVisuals(uv=t_uvs_opt, material=material_td)
                scene.add_geometry(t_mesh, geom_name=f"Terrain_{cx}_{cy}")
                
                if len(bldg_roof_faces) > 0:
                    r_verts_opt, r_faces_opt = optimize_topology(b_verts, bldg_roof_faces, target_reduction=0.7)
                    r_uvs_opt = recalculate_uvs(r_verts_opt, offset_x_world, offset_y_world, scale)
                    b_roof = trimesh.Trimesh(vertices=r_verts_opt, faces=r_faces_opt, process=False)
                    b_roof.visual = trimesh.visual.TextureVisuals(uv=r_uvs_opt, material=material_td)
                    scene.add_geometry(b_roof, geom_name=f"BldgRoof_{cx}_{cy}")
                    
                if len(bldg_wall_faces) > 0:
                    w_verts_opt, w_faces_opt = optimize_topology(b_verts, bldg_wall_faces, target_reduction=0.9) # Paredes retas deciman quase tudo!
                    w_uvs_opt = recalculate_uvs(w_verts_opt, offset_x_world, offset_y_world, scale)
                    b_wall = trimesh.Trimesh(vertices=w_verts_opt, faces=w_faces_opt, process=False)
                    b_wall.visual = trimesh.visual.TextureVisuals(uv=w_uvs_opt, material=material_wall)
                    scene.add_geometry(b_wall, geom_name=f"BldgWall_{cx}_{cy}")
                
        # 4.5 [V5 ARCHITECT] Bake Prefabs into Master Scene
        if sam3_metadata and "objects" in sam3_metadata:
            _log.phase("V5_BAKE").info(f"Integrando {len(sam3_metadata['objects'])} True 3D Objects na malha principal...")
            for obj in sam3_metadata["objects"]:
                glb_path = obj.get("glb_path")
                if not glb_path or not os.path.exists(glb_path):
                    continue
                
                try:
                    # Carrega o modelo isolado sólido gerado pelo Hunyuan3D-2
                    prefab = trimesh.load(glb_path, force='scene')
                    
                    x, y, w, h = obj["bbox"]
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    
                    concept = obj.get("concept_safe", "object")
                    obj_id = obj["id"]
                    
                    # Calcular a posição em World Space 
                    # Usa eff_res=1023.0 conforme padronizado no extract_semantic_prefabs
                    world_x = cx * (scale / 1023.0)
                    world_y = cy * (scale / 1023.0)
                    
                    # Escala básica: o prefab gerado pelo H3D-2 geralmente vem normalizado [-0.5, 0.5]
                    # Ajustamos o tamanho real com base na bounding box original recortada do terreno.
                    real_width = w * (scale / 1023.0)
                    # Estimativa de altura razoável (predios costumam ser mais altos que largos)
                    real_height = real_width * 1.5
                    
                    # Cria a matriz de transformação
                    matrix = trimesh.transformations.scale_matrix(real_width) # Scale is uniform for now
                    # Translada para o lugar exato no mapa
                    trans_matrix = trimesh.transformations.translation_matrix([world_x, 0, world_y]) # Z is 0 (ground level)
                    final_matrix = np.dot(trans_matrix, matrix)
                    
                    # Se prefab for uma Scene (múltiplas geometrias), itera e adiciona
                    if isinstance(prefab, trimesh.Scene):
                        for geom_name, geom in prefab.geometry.items():
                            geom.apply_transform(final_matrix)
                            scene.add_geometry(geom, geom_name=f"{concept}_{obj_id}_{geom_name}")
                    else:
                        prefab.apply_transform(final_matrix)
                        scene.add_geometry(prefab, geom_name=f"{concept}_{obj_id}")
                        
                    _log.phase("V5_BAKE").ok(f"  └ {concept} ({obj_id}) instanciado com sucesso.")
                except Exception as e:
                    _log.phase("V5_BAKE").warn(f"Falha ao instanciar prefab {glb_path}: {e}")

        # 5. Exportar GLB massivo único
        _log.phase("EXPORT").info(f"Exportando cena completa (Terreno + Instâncias): {output_path}")
        scene.export(output_path)
        
        # 6. [JSON FAB SPAWNER]
        extract_semantic_prefabs(scene, sam3_metadata=sam3_metadata or {}, output_dir=base_dir, scale=scale)
        
        _log.phase("EXPORT").ok("Terreno e JSON FAB SPAWNER gerados com sucesso!")
        return True
        
    except Exception as e:
        _log.phase("ERROR").error(f"Falha crítica ao gerar landscape", exc=e)
        return False

# Para testes isolados
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI World Engine - V3 Generator")
    parser.add_argument("--image", required=True, help="Imagem RGB base (Texture)")
    parser.add_argument("--depth", required=True, help="Mapa de Profundidade")
    parser.add_argument("--output", required=True, help="Saída GLB")
    parser.add_argument("--chunk_resolution", type=int, default=1024, help="Resolução de cada lote do grid de VRAM (Ex: 1024)")
    args = parser.parse_args()
    
    generate_landscape_mesh(
        args.image, args.depth, args.output, chunk_resolution=args.chunk_resolution
    )
