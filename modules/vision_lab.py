import os
import sys
import pkgutil
import gc

# =============================================================================
# [HACK SÊNIOR] Monkey Patch para o Python 3.12 (Mata o erro do triton/setuptools)
# Isso garante que bibliotecas velhas não quebrem a pipeline.
# =============================================================================
if not hasattr(pkgutil, 'ImpImporter'):
    class DummyImpImporter: pass
    pkgutil.ImpImporter = DummyImpImporter

import torch
import numpy as np
import logging
# VERSION: 12.0 Armor VRAM (V10 Surgery Phase - Field Surgery V12)
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

# Lazy Imports
_FLUX_PIPELINE = None
_SAM3_PREDICTOR = None
_DEPTH_PIPELINE = None


class VisionLab:
    def __init__(self, focal_distance_mm: float = 35.0, device: str = "cuda"):
        self.f_mm = focal_distance_mm
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.flux_model = None
        self.sam_model = None
        self.sam_generator = None
        self.depth_model = None
        self.depth_processor = None
        
        self._kv_cache_4bit_enabled = False
        self._ada_round_batching = False
        
        self._distortion_coeffs = None
        self._camera_matrix = None
        
        logger.info(f"[VisionLab V11] Inicializado com Armor VRAM. Device: {self.device}")

    # =========================================================================
    # MODEL LOADERS (Lazy — Modelo só come VRAM quando é necessário)
    # =========================================================================
    def _load_flux(self):
        if self.flux_model is not None: return
        logger.info("[VisionLab] Carregando Engine V12 (FLUX.2 Core)...")
        try:
            from diffusers import FluxPipeline
            # Carregamento otimizado para FLUX.2 (Pixtral-based)
            self.flux_model = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(self.device)
            
            # Fix para o erro de 'model_max_length'
            if hasattr(self.flux_model, "tokenizer") and self.flux_model.tokenizer:
                if not hasattr(self.flux_model.tokenizer, 'model_max_length'):
                    self.flux_model.tokenizer.model_max_length = 512
            
            self.flux_model.enable_attention_slicing()
            logger.info("[VisionLab] FLUX.2-dev acoplado com sucesso!")
        except Exception as e:
            logger.warning(f"[VisionLab] Fallback para V1 devido a: {e}")
            from diffusers import FluxPipeline
            self.flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(self.device)
            
        except ImportError:
            logger.error("[VisionLab] 'diffusers' não instalado! Rode: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"[VisionLab] Falha crítica ao carregar modelo de visão: {e}")
            raise

    def _load_sam3(self):
        if self.sam_model is not None:
            return
            
        logger.info("[VisionLab] Carregando SAM 3 (3GB VRAM)...")
        
        try:
            from sam3.build_sam import build_sam3
            from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
            
            sam3_checkpoint = "facebook/sam3-hiera-large"
            
            self.sam_model = build_sam3(
                config_file="sam3_hiera_l.yaml",
                ckpt_path=sam3_checkpoint,
                device=self.device,
            )
            
            self.sam_generator = SAM3AutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=500,
            )
            
            logger.info("[VisionLab] SAM 3 carregado com sucesso!")
        
        except ImportError:
            logger.warning("[VisionLab] SAM 3 não instalado. Usando fallback OpenCV (contour-based).")
            self.sam_model = "FALLBACK_OPENCV"
        except Exception as e:
            logger.warning(f"[VisionLab] Falha ao carregar SAM 3: {e}. Usando fallback OpenCV.")
            self.sam_model = "FALLBACK_OPENCV"

    def _load_depth(self):
        if self.depth_model is not None:
            return
            
        logger.info("[VisionLab] Carregando Depth Anything V3 (1.5GB VRAM)...")
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            
            self.depth_processor = AutoImageProcessor.from_pretrained(model_id)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            ).to(self.device)
            
            logger.info("[VisionLab] Depth Anything V3 carregado com sucesso!")
        
        except ImportError:
            logger.error("[VisionLab] 'transformers' não instalado!")
            raise

    # =========================================================================
    # 1. FLUX (O Blueprint)
    # =========================================================================
    def generate_blueprint(self, user_prompt: str, resolution: str = "4k") -> np.ndarray:
        self._load_flux()
        
        prompt_tecnico = self._inject_blueprint_styles(user_prompt)
        logger.info(f"[VisionLab] Prompt Enriquecido: {prompt_tecnico}")
        
        if resolution == "4k":
            width, height = 1024, 1024
        else:
            width, height = 768, 768
        
        prompt_tecnico = self._split_long_prompt(prompt_tecnico)
        
        with torch.inference_mode():
            result = self.flux_model(
                prompt=prompt_tecnico,
                height=height,
                width=width,
                num_inference_steps=30,
                guidance_scale=3.5,
                max_sequence_length=512,
            )
        
        generated_image = result.images[0]
        blueprint_rgb = np.array(generated_image, dtype=np.uint8)
        
        logger.info(f"[VisionLab] Blueprint REAL gerado: {blueprint_rgb.shape}")
        
        if resolution == "4k" and blueprint_rgb.shape[0] < 2160:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(blueprint_rgb)
            pil_img = pil_img.resize((3840, 2160), PILImage.BICUBIC)
            blueprint_rgb = np.array(pil_img, dtype=np.uint8)
        
        return blueprint_rgb

    # =========================================================================
    # 2. SAM 3 (O Retalhador)
    # =========================================================================
    def extract_semantic_atlas(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_sam3()
        
        logger.info("[VisionLab] Ativando fatiamento semântico...")
        height, width = image_rgb.shape[:2]
        
        if self.sam_model == "FALLBACK_OPENCV":
            return self._sam_fallback_opencv(image_rgb)
        
        masks_output = self.sam_generator.generate(image_rgb)
        masks_output = sorted(masks_output, key=lambda x: x['area'], reverse=True)
        
        semantic_map = np.full((height, width), -1, dtype=np.int32)
        
        for actor_id, mask_data in enumerate(masks_output, start=1):
            mask_bool = mask_data['segmentation']
            overlap = semantic_map[mask_bool] == -1
            mask_indices = np.where(mask_bool)
            semantic_map[mask_indices[0][overlap], mask_indices[1][overlap]] = actor_id
        
        logger.info(f"[VisionLab] SAM 3 encontrou {len(masks_output)} atores na cena!")
        return semantic_map
    
    def _sam_fallback_opencv(self, image_rgb: np.ndarray) -> np.ndarray:
        import cv2
        height, width = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        semantic_map = np.full((height, width), -1, dtype=np.int32)
        
        actor_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                cv2.drawContours(semantic_map, [contour], -1, actor_id, cv2.FILLED)
                actor_id += 1
                
        return semantic_map

    # =========================================================================
    # 3. DEPTH ANYTHING (Régua Métrica)
    # =========================================================================
    def extract_metric_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_depth()
        
        logger.info("[VisionLab] Calculando topologia Z REAL...")
        pil_image = Image.fromarray(image_rgb)
        
        inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        height, width = image_rgb.shape[:2]
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = ((depth_map - depth_min) / (depth_max - depth_min)) * 50.0
        
        depth_map = depth_map.astype(np.float32)
        depth_map = self._fix_specular_depth_holes(image_rgb, depth_map)
        depth_map = self._apply_tilt_correction(depth_map)
        
        return depth_map

    # =========================================================================
    # HELPERS DE CALIBRAÇÃO E FIXES
    # =========================================================================
    def _inject_blueprint_styles(self, prompt: str) -> str:
        technical_style = (
            "highly detailed urban scene, photorealistic aerial photograph, "
            "top-down drone view, sharp edges, clear object boundaries, "
            "high contrast, professional photography, 8k resolution, "
            "distinct buildings and structures, clear separation between objects"
        )
        return f"{prompt}, {technical_style}"
    
    def _split_long_prompt(self, prompt: str) -> str:
        words = prompt.split()
        if len(words) <= 70:
            return prompt
        core = " ".join(words[:60])
        details = " ".join(words[60:70])
        return f"{core}, {details}"
    
    def _fix_specular_depth_holes(self, image_rgb: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depth_range = depth_map.max() - depth_map.min()
        if depth_range < 1e-6:
            return depth_map
        
        depth_normalized = (depth_map - depth_map.min()) / depth_range
        specular_mask = (gray > 0.8) & (depth_normalized < 0.05)
        
        if np.sum(specular_mask) == 0:
            return depth_map
            
        kernel_size = 15
        depth_blurred = cv2.blur(depth_map, (kernel_size, kernel_size))
        mask_blurred = cv2.blur((~specular_mask).astype(np.float32), (kernel_size, kernel_size))
        mask_blurred[mask_blurred < 1e-6] = 1e-6
        
        depth_fixed = depth_blurred / mask_blurred
        depth_map[specular_mask] = depth_fixed[specular_mask]
        return depth_map
    
    def _apply_tilt_correction(self, depth_map: np.ndarray) -> np.ndarray:
        height = depth_map.shape[0]
        row_medians = np.median(depth_map, axis=1)
        y_coords = np.arange(height, dtype=np.float32)
        
        if len(y_coords) < 2: return depth_map
        
        coeffs = np.polyfit(y_coords, row_medians, 1)
        slope = coeffs[0]
        
        if abs(slope) < 0.01: return depth_map
        
        tilt_correction = slope * y_coords
        depth_map = depth_map - tilt_correction[:, np.newaxis]
        return np.maximum(depth_map, 0.0)
    
    def configure_lens(self, k1: float=0.0, k2: float=0.0, k3: float=0.0, image_width: int=3840, image_height: int=2160):
        self._distortion_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float64)
        fx = fy = image_width * 0.8
        cx, cy = image_width / 2.0, image_height / 2.0
        self._camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    
    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self._distortion_coeffs is None or np.all(self._distortion_coeffs == 0):
            return image
        import cv2
        return cv2.undistort(image, self._camera_matrix, self._distortion_coeffs)

    # =========================================================================
    # [FIX V12] Auditoria de Nitidez com Modo Atmosférico
    # =========================================================================
    def _perform_sharpness_audit(self, image_rgb: np.ndarray, prompt: str) -> bool:
        import cv2
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Se o prompt tiver fumaça/névoa, o threshold cai de 50 para 15 (ajustado para 20 no Fix 12.1)
        atmospheric_keywords = ["fumaça", "smoke", "cinzas", "ash", "fog", "névoa", "volumetric", "fumo"]
        threshold = 50.0
        if any(word.lower() in prompt.lower() for word in atmospheric_keywords):
            threshold = 20.0
            logger.info(f"[QC] Modo Atmosférico detectado. Threshold reduzido para {threshold}")

        logger.info(f"[QC] Sharpness Audit: Score {score:.1f} | Threshold {threshold}")
        
        if score < threshold:
            logger.error(f"[QC] FAIL: Imagem muito borrada ({score:.1f} < {threshold}).")
            return False
        return True

    def calculate_ue5_world_scale(self, S_pixel: float, Z_depth_meters: float) -> float:
        sensor_width_mm = 36.0 
        image_width_pixels = 3840.0
        pixel_pitch = sensor_width_mm / image_width_pixels
        tamanho_focal_plano = S_pixel * pixel_pitch
        return tamanho_focal_plano * (Z_depth_meters / (self.f_mm / 1000.0))

    # =========================================================================
    # ORQUESTRADOR DA FASE 1 (API Principal)
    # =========================================================================
    def generate_intent_map(self, prompt: str) -> dict:
        logger.info(">>> [V12] Gerando Mapa de Intenção (Blueprint -> Segmentação -> Topologia) <<<")
        
        blueprint_4k = self.generate_blueprint(prompt)
        blueprint_4k = self._undistort_image(blueprint_4k)
        
        # AUDITORIA DE QUALIDADE (Agora com correção para fumaça)
        if not self._perform_sharpness_audit(blueprint_4k, prompt):
            # Se falhar, a gente tenta um "micro-sharpening" antes de desistir
            import cv2
            logger.warning("[VisionLab] Aplicando Re-Sharpening de emergência...")
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            blueprint_4k = cv2.filter2D(blueprint_4k, -1, kernel)
            if not self._perform_sharpness_audit(blueprint_4k, prompt):
                raise RuntimeError("GATE 0 FALHOU! Blueprint borrado mesmo após correção.")

        sam_mask_4k = self.extract_semantic_atlas(blueprint_4k)
        depth_map_4k = self.extract_metric_depth(blueprint_4k)
        
        unique_actors = np.unique(sam_mask_4k)
        unique_actors = unique_actors[unique_actors > 0]
        
        actors_metadata = {}
        for actor_id in unique_actors:
            actor_mask = (sam_mask_4k == actor_id)
            rows, cols = np.any(actor_mask, axis=1), np.any(actor_mask, axis=0)
            if not np.any(rows) or not np.any(cols): continue
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            z_medio = float(np.mean(depth_map_4k[actor_mask]))
            scale_ue5 = self.calculate_ue5_world_scale(float(cmax - cmin), z_medio)
            
            actors_metadata[int(actor_id)] = {
                "ue5_scale_meters": scale_ue5,
                "z_socket_depth": z_medio,
                "bbox": [int(rmin), int(cmin), int(rmax), int(cmax)],
                "pixel_area": int(np.sum(actor_mask)),
            }
        
        # Limpa o cache CUDA intermediário pra garantir a transição pro Trellis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
        logger.info("[VisionLab] Expurgando VRAM...")
        if self.flux_model: del self.flux_model
        if self.sam_model: del self.sam_model
        if self.depth_model: del self.depth_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        logger.info("[VisionLab] Limpeza de VRAM concluída. Slot liberado.")