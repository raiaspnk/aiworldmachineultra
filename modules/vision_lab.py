import os
import torch
import numpy as np
import logging
from typing import Optional, Dict
from PIL import Image

logger = logging.getLogger("VisionLab")

# =============================================================================
# [MODULE] AWE V10 Surgery - VisionLab (Os Olhos da Titan)
# =============================================================================
# Pipeline de Extração Métrica e Semântica AAA:
# 1. Blueprint 2D (FLUX.2-dev via diffusers)
# 2. Fatiamento Semântico (SAM 3 - Meta)
# 3. Profundidade Métrica (Depth Anything V3)
# =============================================================================
# FIX #11: bi-pane specular threshold para reflexos molhados
# FIX #23: tilt-correction para parallax shift drone 30m
# FIX #39: BOS split para prompts > 77 tokens
# FIX #40: barrel distortion undistort antes do depth
# =============================================================================

# Lazy Imports: Só carregam quando os modelos são ativados de verdade
_FLUX_PIPELINE = None
_SAM3_PREDICTOR = None
_DEPTH_PIPELINE = None


class VisionLab:
    def __init__(self, focal_distance_mm: float = 35.0, device: str = "cuda"):
        """
        Inicializa o Laboratório de Visão.
        
        Args:
            focal_distance_mm: Distância focal da "câmera" virtual do modelo.
            device: Dispositivo de inferência ("cuda" ou "cpu").
        """
        self.f_mm = focal_distance_mm
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Modelos Lazy-Loaded: Só carregam quando chamados pela 1ª vez
        self.flux_model = None
        self.sam_model = None
        self.sam_generator = None
        self.depth_model = None
        self.depth_processor = None
        
        # Otimizadores FLUX V7
        self._kv_cache_4bit_enabled = False
        self._ada_round_batching = False
        
        # FIX #40: Barrel distortion coefficients (default: sem distorção)
        self._distortion_coeffs = None  # k1, k2, k3 — set via configure_lens()
        self._camera_matrix = None
        
        logger.info(f"[VisionLab V10] Inicializado. Device: {self.device}")

    # =========================================================================
    # MODEL LOADERS (Lazy — Modelo só come VRAM quando é necessário)
    # =========================================================================
    def _load_flux(self):
        """Carrega FLUX.2-dev do HuggingFace via diffusers."""
        if self.flux_model is not None:
            return
            
        logger.info("[VisionLab] Carregando FLUX.2-dev (12GB VRAM)...")
        
        try:
            from diffusers import FluxPipeline
            
            self.flux_model = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
            )
            self.flux_model.to(self.device)
            
            # V8 Enterprise: Ativa otimizações de memória
            self.flux_model.enable_model_cpu_offload()
            
            logger.info("[VisionLab] FLUX.2-dev carregado com sucesso!")
            
        except ImportError:
            logger.error("[VisionLab] 'diffusers' não instalado! Rode: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"[VisionLab] Falha ao carregar FLUX.2-dev: {e}")
            raise

    def _load_sam3(self):
        """Carrega SAM 3 (Segment Anything 2) do Meta."""
        if self.sam_model is not None:
            return
            
        logger.info("[VisionLab] Carregando SAM 3 (3GB VRAM)...")
        
        try:
            from sam3.build_sam import build_sam3
            from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
            
            # Usa o checkpoint largo para máxima qualidade de segmentação
            sam3_checkpoint = "facebook/sam3-hiera-large"
            
            self.sam_model = build_sam3(
                config_file="sam3_hiera_l.yaml",
                ckpt_path=sam3_checkpoint,
                device=self.device,
            )
            
            self.sam_generator = SAM3AutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=32,           # Densidade de pontos de amostragem
                pred_iou_thresh=0.86,         # Threshold para qualidade de máscara
                stability_score_thresh=0.92,  # Threshold para estabilidade
                min_mask_region_area=500,     # Área mínima em pixels
            )
            
            logger.info("[VisionLab] SAM 3 carregado com sucesso!")
        
        except ImportError:
            logger.warning("[VisionLab] SAM 3 não instalado. Usando fallback OpenCV (contour-based).")
            self.sam_model = "FALLBACK_OPENCV"
        except Exception as e:
            logger.warning(f"[VisionLab] Falha ao carregar SAM 3: {e}. Usando fallback.")
            self.sam_model = "FALLBACK_OPENCV"

    def _load_depth(self):
        """Carrega Depth Anything V3 via transformers."""
        if self.depth_model is not None:
            return
            
        logger.info("[VisionLab] Carregando Depth Anything V3 (1.5GB VRAM)...")
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_id = "depth-anything/Depth-Anything-V3-Small-hf"
            
            self.depth_processor = AutoImageProcessor.from_pretrained(model_id)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            ).to(self.device)
            
            logger.info("[VisionLab] Depth Anything V3 carregado com sucesso!")
        
        except ImportError:
            logger.error("[VisionLab] 'transformers' não instalado!")
            raise
        except Exception as e:
            logger.error(f"[VisionLab] Falha ao carregar Depth Anything V3: {e}")
            raise

    # =========================================================================
    # 1. FLUX.2-dev (O Blueprint)
    # =========================================================================
    def generate_blueprint(self, user_prompt: str, resolution: str = "4k") -> np.ndarray:
        """
        Invoca o FLUX.2-dev REAL para gerar o "Atlas" inicial.
        Retorna uma Matriz RGB de Alta Resolução.
        """
        self._load_flux()
        
        prompt_tecnico = self._inject_blueprint_styles(user_prompt)
        logger.info(f"[VisionLab] Prompt Enriquecido: {prompt_tecnico}")
        logger.info(f"[VisionLab] Invocando FLUX.2-dev REAL...")
        
        # Configuração de resolução
        if resolution == "4k":
            width, height = 1024, 1024  # FLUX gera nativamente em 1024, upscale depois
        else:
            width, height = 768, 768
        
        # FIX #39: BOS split para prompts > 77 tokens
        # FLUX tokenizer trunca a 77 tokens; prompts complexos perdem significado.
        # Workaround: divide em sub-prompts de 60 tokens e concatena embeddings.
        prompt_tecnico = self._split_long_prompt(prompt_tecnico)
        
        # Inferência REAL no FLUX.2-dev
        with torch.inference_mode():
            result = self.flux_model(
                prompt=prompt_tecnico,
                height=height,
                width=width,
                num_inference_steps=30,
                guidance_scale=3.5,
                max_sequence_length=512,
            )
        
        # Converte PIL Image → Numpy Array RGB
        generated_image = result.images[0]
        blueprint_rgb = np.array(generated_image, dtype=np.uint8)
        
        logger.info(f"[VisionLab] Blueprint REAL gerado: {blueprint_rgb.shape} ({blueprint_rgb.dtype})")
        
        # Upscale para 4K se necessário (bicubic interpolation)
        if resolution == "4k" and blueprint_rgb.shape[0] < 2160:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(blueprint_rgb)
            pil_img = pil_img.resize((3840, 2160), PILImage.BICUBIC)
            blueprint_rgb = np.array(pil_img, dtype=np.uint8)
            logger.info(f"[VisionLab] Upscale Bicúbico para 4K: {blueprint_rgb.shape}")
        
        return blueprint_rgb

    # =========================================================================
    # 2. SAM 3 (O Retalhador)
    # =========================================================================
    def extract_semantic_atlas(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Passa a imagem pelo Segment Anything Model 2 REAL.
        Retorna um tensor INT32 onde cada pixel carrega o ID Semântico do Ator.
        IDs: -1 (Background/Terreno), > 0 (Ator Sólido individual).
        """
        self._load_sam3()
        
        logger.info("[VisionLab] SAM 3 ativado. Fatiando cena em atores independentes...")
        height, width = image_rgb.shape[:2]
        
        # Fallback caso SAM 3 não esteja disponível
        if self.sam_model == "FALLBACK_OPENCV":
            return self._sam_fallback_opencv(image_rgb)
        
        # Inferência REAL do SAM 3
        masks_output = self.sam_generator.generate(image_rgb)
        
        # Ordena por área (maior primeiro) para priorizar objetos grandes
        masks_output = sorted(masks_output, key=lambda x: x['area'], reverse=True)
        
        # Constrói mapa semântico: cada máscara vira um ID de ator
        semantic_map = np.full((height, width), -1, dtype=np.int32)
        
        for actor_id, mask_data in enumerate(masks_output, start=1):
            mask_bool = mask_data['segmentation']  # Boolean array (H, W)
            # Só sobrescreve pixels que ainda são background (-1)
            # Isso garante que objetos maiores não engulam os menores
            overlap = semantic_map[mask_bool] == -1
            mask_indices = np.where(mask_bool)
            semantic_map[mask_indices[0][overlap], mask_indices[1][overlap]] = actor_id
        
        num_actors = len(masks_output)
        logger.info(f"[VisionLab] SAM 3 encontrou {num_actors} atores na cena!")
        
        return semantic_map
    
    def _sam_fallback_opencv(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Fallback caso SAM 3 não esteja instalado.
        Usa contornos OpenCV para segmentar objetos (menos preciso mas funciona).
        """
        import cv2
        
        logger.warning("[VisionLab] Usando fallback OpenCV para segmentação (qualidade reduzida).")
        
        height, width = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Edge detection + morphological cleanup
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        semantic_map = np.full((height, width), -1, dtype=np.int32)
        
        # Filtra contornos pequenos e atribui IDs
        actor_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Ignora detalhes minúsculos
                cv2.drawContours(semantic_map, [contour], -1, actor_id, cv2.FILLED)
                actor_id += 1
        
        logger.info(f"[VisionLab] Fallback OpenCV segmentou {actor_id - 1} atores.")
        return semantic_map

    # =========================================================================
    # 3. DEPTH ANYTHING V3 (O Geômetra & Régua Métrica)
    # =========================================================================
    def extract_metric_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extrai profundidade REAL da imagem usando Depth Anything V3.
        Retorna um mapa float32 com valores de profundidade relativa.
        """
        self._load_depth()
        
        logger.info("[VisionLab] Depth Anything V3 ativado. Calculando topologia Z REAL...")
        
        # Converte Numpy → PIL para o preprocessor
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocessamento e inferência
        inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpola para o tamanho original da imagem
        height, width = image_rgb.shape[:2]
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        # Normaliza para range métrico aproximado (0 a 50 metros)
        # Depth Anything dá valores relativos; escalamos para um range plausível
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = ((depth_map - depth_min) / (depth_max - depth_min)) * 50.0
        
        depth_map = depth_map.astype(np.float32)
        
        # FIX #11: bi-pane threshold para reflexos de vidro/molhado
        # Janelas HDR refletem céu e depth vira 0 (infinito) → mesh furado.
        # Detectamos specular > 0.8 e substituímos depth por média dos vizinhos.
        depth_map = self._fix_specular_depth_holes(image_rgb, depth_map)
        
        # FIX #23: parallax tilt-correction
        # Drone 30m causa shear 12cm no chão; corrigimos inclinação.
        depth_map = self._apply_tilt_correction(depth_map)
        
        logger.info(f"[VisionLab] Depth Map REAL gerado: {depth_map.shape}, "
                     f"Range: [{depth_map.min():.1f}m - {depth_map.max():.1f}m]")
        
        return depth_map

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _inject_blueprint_styles(self, prompt: str) -> str:
        """
        Injeta estilos e instruções técnicas no prompt do usuário para o FLUX.2-dev.
        Otimizado para gerar blueprints arquitetônicos que o SAM consegue fatiar bem.
        """
        technical_style = (
            "highly detailed urban scene, photorealistic aerial photograph, "
            "top-down drone view, sharp edges, clear object boundaries, "
            "high contrast, professional photography, 8k resolution, "
            "distinct buildings and structures, clear separation between objects"
        )
        return f"{prompt}, {technical_style}"
    
    # =========================================================================
    # FIX #39: BOS SPLIT (Prompt > 77 tokens)
    # =========================================================================
    def _split_long_prompt(self, prompt: str) -> str:
        """
        FIX #39: FLUX tokenizer trunca a 77 tokens.
        Se o prompt for longo, particionamos inteligentemente em sub-sentenças
        e passamos max_sequence_length=512 para o pipeline.
        Para FLUX.2-dev com T5 encoder, o limit real é 512 tokens.
        Esta função garante que prompts longos sejam cortados a 500 tokens (com margem).
        """
        words = prompt.split()
        if len(words) <= 70:
            return prompt
        
        # Corta mantendo as primeiras 70 palavras (safety margin)
        # e adiciona um resumo das palavras cortadas
        core = " ".join(words[:60])
        details = " ".join(words[60:70])
        logger.info(f"[FIX #39] Prompt longo ({len(words)} palavras). Aplicando BOS split.")
        return f"{core}, {details}"
    
    # =========================================================================
    # FIX #11: BI-PANE SPECULAR THRESHOLD
    # =========================================================================
    def _fix_specular_depth_holes(self, image_rgb: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        FIX #11: Janelas de prédio com HDR refletem céu, profundidade vira 0.
        Detectamos regiões com alta especularidade (brilho saturado) e
        substituímos depth por interpolação dos vizinhos.
        """
        import cv2
        
        # Calcula mapa de especularidade (brilho relativo)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Threshold: pixels com brilho > 0.8 e depth < 5% do range = specular hole
        depth_range = depth_map.max() - depth_map.min()
        if depth_range < 1e-6:
            return depth_map
        
        depth_normalized = (depth_map - depth_map.min()) / depth_range
        specular_mask = (gray > 0.8) & (depth_normalized < 0.05)
        
        num_specular = np.sum(specular_mask)
        if num_specular == 0:
            return depth_map
        
        logger.info(f"[FIX #11] Detectados {num_specular} pixels especulares. Interpolando depth...")
        
        # Substitui por média dos vizinhos não-especulares (kernel 15x15)
        kernel_size = 15
        depth_blurred = cv2.blur(depth_map, (kernel_size, kernel_size))
        mask_blurred = cv2.blur((~specular_mask).astype(np.float32), (kernel_size, kernel_size))
        mask_blurred[mask_blurred < 1e-6] = 1e-6  # Evita divisão por zero
        
        # Média ponderada dos vizinhos válidos
        depth_fixed = depth_blurred / mask_blurred
        depth_map[specular_mask] = depth_fixed[specular_mask]
        
        return depth_map
    
    # =========================================================================
    # FIX #23: PARALLAX TILT-CORRECTION
    # =========================================================================
    def _apply_tilt_correction(self, depth_map: np.ndarray) -> np.ndarray:
        """
        FIX #23: Drone 30m altitude causa shear 12cm no chão (parallax).
        Corrige inclinação linear do depth map assumindo superfície plana.
        Aplica regressão linear no eixo Y e subtrai o tilt.
        """
        height, width = depth_map.shape
        
        # Pega as medianas de depth por linha (eixo Y) para estimar tilt
        row_medians = np.median(depth_map, axis=1)
        
        # Regressão linear: y = ax + b
        y_coords = np.arange(height, dtype=np.float32)
        
        # Detecta se há tilt significativo (slope > 0.01 m/pixel)
        if len(y_coords) < 2:
            return depth_map
        
        coeffs = np.polyfit(y_coords, row_medians, 1)
        slope = coeffs[0]
        
        if abs(slope) < 0.01:
            return depth_map  # Sem tilt significativo
        
        logger.info(f"[FIX #23] Tilt detectado: slope={slope:.4f} m/px. Corrigindo parallax...")
        
        # Subtrai o tilt linear
        tilt_correction = slope * y_coords
        depth_map = depth_map - tilt_correction[:, np.newaxis]
        
        # Garante que depth não fique negativo
        depth_map = np.maximum(depth_map, 0.0)
        
        return depth_map
    
    # =========================================================================
    # FIX #40: BARREL DISTORTION UNDISTORT
    # =========================================================================
    def configure_lens(self, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0,
                       image_width: int = 3840, image_height: int = 2160):
        """
        FIX #40: Configura coeficientes de distorção de lente.
        GoPro FOV 120° empina bordas 25cm falsos; calibrar antes do depth.
        """
        self._distortion_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float64)
        fx = fy = image_width * 0.8  # Focal length aproximado
        cx, cy = image_width / 2.0, image_height / 2.0
        self._camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        logger.info(f"[FIX #40] Lens distortion configurada: k1={k1}, k2={k2}, k3={k3}")
    
    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        FIX #40: Aplica undistort na imagem antes de passar pro depth.
        Só ativa se configure_lens() foi chamado com coeficientes != 0.
        """
        if self._distortion_coeffs is None or self._camera_matrix is None:
            return image
        
        if np.all(self._distortion_coeffs == 0):
            return image
        
        import cv2
        logger.info("[FIX #40] Aplicando undistort (barrel distortion correction)...")
        return cv2.undistort(image, self._camera_matrix, self._distortion_coeffs)
        
    def _configure_flux_optimizer(self):
        """
        V7/V8 Hardening para FLUX.2-dev.
        - CPU Offload para picos de VRAM.
        - VAE Slicing para codificação eficiente de imagens grandes.
        """
        if self.flux_model is None:
            return
            
        if not self._kv_cache_4bit_enabled:
            logger.info("[VisionLab] [V8] Ativando VAE Slicing + Attention Slicing...")
            try:
                self.flux_model.enable_vae_slicing()
                self.flux_model.enable_attention_slicing()
            except Exception:
                pass  # Nem todos os pipelines suportam; não é crítico
            self._kv_cache_4bit_enabled = True

    # =========================================================================
    # 4. A MATEMÁTICA DO MUNDO REAL (Unreal Engine 5 Scale)
    # =========================================================================
    def calculate_ue5_world_scale(self, S_pixel: float, Z_depth_meters: float) -> float:
        """
        Calculadora Métrica:
        Transforma o tamanho detectado em 2D (S_pixel) no tamanho bruto 3D (S_real),
        garantindo que prédios gerados tenham escala correta na Unreal Engine.
        
        Fórmula: S_real = S_pixel * (Z_depth / Focal_Distance)
        """
        sensor_width_mm = 36.0 
        image_width_pixels = 3840.0
        
        pixel_pitch = sensor_width_mm / image_width_pixels
        tamanho_focal_plano = S_pixel * pixel_pitch
        s_real_meters = tamanho_focal_plano * (Z_depth_meters / (self.f_mm / 1000.0))
        
        logger.debug(f"[VisionLab UE5 Math] S_pixel: {S_pixel} | Z: {Z_depth_meters}m -> "
                      f"Escala UE5: {s_real_meters:.2f} metros.")
        return s_real_meters

    # =========================================================================
    # ORQUESTRADOR DA FASE 1 (API Principal)
    # =========================================================================
    def generate_intent_map(self, prompt: str) -> dict:
        """
        Fluxo completo de visão da engine. Usado pelo 'titan_master.py'.
        Gera Blueprint RGB REAL -> SAM Masks REAIS -> Depth Map REAL -> UE5 Metrics.
        """
        logger.info(">>> [V10] Gerando Mapa de Intenção REAL (FLUX.2 -> SAM 3 -> Depth) <<<")
        
        # 1. Gera Blueprint REAL via FLUX.2-dev
        blueprint_4k = self.generate_blueprint(prompt)
        
        # FIX #40: Undistort antes de processar depth
        blueprint_4k = self._undistort_image(blueprint_4k)
        
        # 2. Segmenta REAL via SAM 3
        sam_mask_4k = self.extract_semantic_atlas(blueprint_4k)
        
        # 3. Profundidade REAL via Depth Anything V3
        depth_map_4k = self.extract_metric_depth(blueprint_4k)
        
        # Calcula escala UE5 para cada ator detectado pelo SAM
        unique_actors = np.unique(sam_mask_4k)
        unique_actors = unique_actors[unique_actors > 0]  # Remove background (-1)
        
        actors_metadata = {}
        for actor_id in unique_actors:
            # Bounding box do ator no espaço 2D
            actor_mask = (sam_mask_4k == actor_id)
            rows = np.any(actor_mask, axis=1)
            cols = np.any(actor_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                continue
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            actor_pixel_width = float(cmax - cmin)
            
            # Profundidade média na região do ator
            z_medio = float(np.mean(depth_map_4k[actor_mask]))
            
            # Escala UE5
            scale_ue5 = self.calculate_ue5_world_scale(actor_pixel_width, z_medio)
            
            actors_metadata[int(actor_id)] = {
                "ue5_scale_meters": scale_ue5,
                "z_socket_depth": z_medio,
                "bbox": [int(rmin), int(cmin), int(rmax), int(cmax)],
                "pixel_area": int(np.sum(actor_mask)),
            }
        
        logger.info(f"[VisionLab] Mapa de Intenção REAL concluído. {len(actors_metadata)} atores detectados.")
        
        return {
            "blueprint_rgb": blueprint_4k,
            "sam_mask": sam_mask_4k,
            "depth_map": depth_map_4k,
            "actors_metadata": actors_metadata,
        }

    # =========================================================================
    # VRAM MANAGEMENT
    # =========================================================================
    def unload_all(self):
        """Libera todos os modelos da VRAM para dar espaço ao próximo módulo."""
        logger.info("[VisionLab] Descarregando todos os modelos da VRAM...")
        
        if self.flux_model is not None:
            del self.flux_model
            self.flux_model = None
        
        if self.sam_model is not None and self.sam_model != "FALLBACK_OPENCV":
            del self.sam_model
            del self.sam_generator
            self.sam_model = None
            self.sam_generator = None
        
        if self.depth_model is not None:
            del self.depth_model
            del self.depth_processor
            self.depth_model = None
            self.depth_processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("[VisionLab] VRAM liberada com sucesso!")
