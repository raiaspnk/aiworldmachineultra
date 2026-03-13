import numpy as np
import cv2
import logging
import json

logger = logging.getLogger("SplineExtractor")

# =============================================================================
# [MODULE] AWE V6 Titan - Spline Extractor (O Eletricista)
# =============================================================================
# MISSÃO: Fios, cabos, antenas e postes NÃO DEVEM ser gerados como polígonos
# pela IA. Eles ficam grossos, derretidos, com cara de salsicha.
#
# A SOLUÇÃO: Extraímos as linhas finas como Splines matemáticas (pontos 3D) 
# e exportamos para a Unreal Engine, que renderiza os cabos com o seu 
# CableComponent nativo — perfeitos, finos e com física real.
# =============================================================================

class SplineExtractor:
    def __init__(self):
        logger.info("[SplineExtractor] Modo Eletricista Ativado. Detectando fios e cabos.")
    
    # -------------------------------------------------------------------------
    # [ETAPA 1] DETECÇÃO DE LINHAS FINAS (Edge Detection + Hough Lines)
    # -------------------------------------------------------------------------
    def detect_wire_candidates(self, image_rgb: np.ndarray, mask_sky: np.ndarray = None) -> list:
        """
        Identifica os candidatos a fios/cabos na imagem do Flux.1.
        
        Lógica:
        1. Converte pra cinza e aplica Canny Edge Detection
        2. Usa Hough Line Transform Probabilístico pra pegar apenas linhas longas e finas
        3. Filtra: Linhas em áreas de "Céu" (SAM mask) têm prioridade (cabos ficam no ar)
        
        Args:
            image_rgb: Blueprint 4K gerado pelo Flux
            mask_sky: Máscara do SAM onde label == "Sky" (prioriza cabos aéreos)
            
        Returns:
            Lista de segmentos de linha [(x1,y1,x2,y2), ...]
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Blur suave pra eliminar ruído mas preservar bordas longas
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny com thresholds agressivos (queremos só linhas fortes)
        edges = cv2.Canny(blurred, 80, 200, apertureSize=3)
        
        # Se temos a máscara do céu, filtramos apenas as edges nessa região
        if mask_sky is not None:
            sky_dilated = cv2.dilate((mask_sky > 0).astype(np.uint8) * 255, None, iterations=5)
            edges = cv2.bitwise_and(edges, sky_dilated)
        
        # Hough Lines Probabilístico (detecta segmentos, não retas infinitas)
        # minLineLength: Ignora linhas menores que 80px (ruído)
        # maxLineGap: Conecta pedaços que estejam a até 15px um do outro
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=60, 
                                 minLineLength=80, maxLineGap=15)
        
        if lines is None:
            logger.info("[SplineExtractor] Nenhum cabo detectado na imagem.")
            return []
            
        # Filtrar: Apenas linhas quase horizontais ou em arco suave (cabos não são verticais)
        wire_candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Ângulo da linha
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # Cabos elétricos raramente são verticais puros (> 75 graus)
            # Postes são verticais, então filtramos eles fora
            if angle < 75:
                wire_candidates.append((int(x1), int(y1), int(x2), int(y2)))
                
        logger.info(f"[SplineExtractor] {len(wire_candidates)} candidatos a cabo detectados.")
        return wire_candidates
    
    # -------------------------------------------------------------------------
    # [ETAPA 2] PROJEÇÃO 3D (Pixel 2D + Depth -> Ponto 3D no Mundo)
    # -------------------------------------------------------------------------
    def project_to_3d_spline(self, wire_segments: list, depth_map: np.ndarray, ue5_scale: float = 1.0, num_subdivisions: int = 10) -> list:
        """
        Converte segmentos de linha 2D em Splines 3D usando o Mapa de Profundidade.
        
        Cada segmento vira uma curva com N pontos de controle no espaço 3D,
        permitindo que a Unreal Engine gere um CableComponent físico perfeito.
        
        Args:
            wire_segments: Lista de [(x1,y1,x2,y2)] do Hough Lines
            depth_map: Mapa de profundidade do Depth Anything V3
            ue5_scale: Fator de escala (pixels para metros UE5)
            num_subdivisions: Quantos pontos de controle por cabo
            
        Returns:
            Lista de splines, cada uma com pontos 3D [(x,y,z), ...]
        """
        map_h, map_w = depth_map.shape[:2]
        splines_3d = []
        
        for seg_idx, (x1, y1, x2, y2) in enumerate(wire_segments):
            control_points = []
            
            for i in range(num_subdivisions + 1):
                t = i / num_subdivisions
                
                # Interpolação linear ao longo do segmento 2D
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                
                # Clamp nas bordas da imagem
                px = max(0, min(px, map_w - 1))
                py = max(0, min(py, map_h - 1))
                
                # Lê a profundidade naquele pixel exato
                z = float(depth_map[py, px])
                
                # Converte tudo pra metros UE5
                world_x = px * ue5_scale
                world_y = py * ue5_scale
                world_z = z * ue5_scale
                
                # Aplica Catenária simulada (cabos pendurados fazem curva por gravidade)
                # A fórmula é parabólica: o ponto mais baixo no centro do cabo
                sag = 0.5 * ue5_scale # 50cm de "barriga" no centro
                catenary_offset = sag * (4.0 * t * (1.0 - t)) # Parábola: máximo em t=0.5
                world_z -= catenary_offset
                
                control_points.append({
                    "x": round(world_x, 3),
                    "y": round(world_y, 3),
                    "z": round(world_z, 3)
                })
            
            splines_3d.append({
                "spline_id": seg_idx,
                "control_points": control_points,
                "type": "CABLE",
                "ue5_component": "CableComponent",
                "physics_enabled": True
            })
        
        logger.info(f"[SplineExtractor] {len(splines_3d)} splines 3D geradas com catenária gravitacional.")
        return splines_3d
    
    # -------------------------------------------------------------------------
    # [ETAPA 3] EXPORTAÇÃO JSON (Para a Unreal Engine importar)
    # -------------------------------------------------------------------------
    def export_splines_json(self, splines_3d: list, output_path: str = "output/splines_world.json"):
        """
        Salva as splines num JSON legível pela Unreal Engine (ou script Python da UE5).
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        payload = {
            "engine": "AWE_V6_Titan",
            "module": "SplineExtractor",
            "total_cables": len(splines_3d),
            "splines": splines_3d
        }
        
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
            
        logger.info(f"[SplineExtractor] Exportado {len(splines_3d)} cabos para '{output_path}'")
        return output_path

    # -------------------------------------------------------------------------
    # PIPELINE PRINCIPAL (API de Entrada)
    # -------------------------------------------------------------------------
    def extract_all_cables(self, image_rgb: np.ndarray, depth_map: np.ndarray, ue5_scale: float, mask_sky: np.ndarray = None) -> list:
        """
        Workflow completo: Detecta -> Projeta 3D -> Retorna Splines prontas.
        """
        logger.info("\n========== [SPLINE EXTRACTOR] Varrendo a Imagem em Busca de Cabos ==========")
        
        # 1. Detecta linhas finas
        wires_2d = self.detect_wire_candidates(image_rgb, mask_sky)
        
        if not wires_2d:
            logger.info("[SplineExtractor] Nenhum cabo encontrado. Pulando.")
            return []
        
        # 2. Projeta no espaço 3D com gravidade simulada
        splines = self.project_to_3d_spline(wires_2d, depth_map, ue5_scale)
        
        # 3. Exporta JSON automaticamente
        self.export_splines_json(splines)
        
        return splines
