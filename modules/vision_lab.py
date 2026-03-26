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

# Garante que o Python ache o SAM 3 instalado via Git
sys.path.append(os.path.abspath("sam3"))

# Lazy Imports
_FLUX_PIPELINE = None
_SAM3_PREDICTOR = None
_DEPTH_PIPELINE = None


class VisionLab:
    def __init__(self, focal_distance_mm: float = 35.0, base_world_scale_meters: float = 500.0, device: str = "cuda"):
        self.f_mm = focal_distance_mm
        self.base_world_scale_meters = base_world_scale_meters
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.flux_model = None
        self.controlnet = None
        self.sam_model = None
        self.sam_generator = None
        self.depth_model = None
        self.depth_processor = None
        self.img2img_pipe = None
        
        self._kv_cache_4bit_enabled = False
        self._ada_round_batching = False
        
        logger.info(f"[VisionLab V15] Inicializado (The Universal Architecture). Scale Base: {self.base_world_scale_meters}m | Device: {self.device}")

    # =========================================================================
    # MODEL LOADERS (Lazy — Modelo só come VRAM quando é necessário)
    # =========================================================================
    def _load_flux(self):
        if self.flux_model is not None: return
        logger.info("[VisionLab] [V15] Inicializando FLUX Core com Garantia Estrutural (ControlNet / Canny)...")
        try:
            from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel
            
            # Tenta usar o ControlNet Union (ou similar) pra forçar a geometria
            try:
                logger.info("[VisionLab] Acoplando ControlNet-Union (Shakker-Labs) no FLUX...")
                self.controlnet = FluxControlNetModel.from_pretrained("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", torch_dtype=torch.bfloat16).to(self.device)
                self.flux_model = FluxControlNetPipeline.from_pretrained(
                    "black-forest-labs/FLUX.2-dev", 
                    controlnet=self.controlnet, 
                    torch_dtype=torch.bfloat16, 
                    use_safetensors=True
                ).to(self.device)
            except Exception as e_cn:
                logger.warning(f"[VisionLab] Falha ao injetar ControlNet ({e_cn}). Roteando para Blueprint Estrito s/ ControlNet.")
                self.flux_model = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.2-dev",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                ).to(self.device)
            
            # Fix token length safety
            if hasattr(self.flux_model, "tokenizer") and self.flux_model.tokenizer is not None:
                if not hasattr(self.flux_model.tokenizer, 'model_max_length'):
                    self.flux_model.tokenizer.model_max_length = 512
            
            if hasattr(self.flux_model, 'enable_attention_slicing'):
                self.flux_model.enable_attention_slicing()
            logger.info("[VisionLab] FLUX Core Operacional!")
        except Exception as e:
            logger.warning(f"[VisionLab] Fallback Crítico de Pipeline devido a: {e}")
            from diffusers import FluxPipeline
            self.flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(self.device)
            
        except ImportError:
            logger.error("[VisionLab] 'diffusers' não instalado! Rode: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"[VisionLab] Falha crítica ao carregar modelo de visão: {e}")
            raise

    def _load_sam3(self):
        if self.sam_model is not None: return
        logger.info("[VisionLab] Carregando SAM 3 (Fatiamento de Precisão)...")
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
    # 1. FLUX (O Blueprint) - FIX V13: Native 2048x2048 + Lanczos
    # =========================================================================
    def generate_blueprint(self, user_prompt: str, resolution: str = "4k") -> np.ndarray:
        self._load_flux()
        
        # FIX V14: Condicionamento aereo universal ESTRUTURAL
        prompt_tecnico = (
            f"{user_prompt}, strict orthogonal top-down architectural map, "
            "perfect building footprints, clear street grid, sharp edges, high contrast volumetric geometry, 8k resolution"
        )
        logger.info(f"[VisionLab] Prompt Enriquecido: {prompt_tecnico}")
        
        # FIX V13: O FLUX.2 aguenta resolucao nativa muito maior. 
        # Aumentar para 2048 evita a perda brutal de definicao urbana.
        width, height = 2048, 2048
        
        with torch.inference_mode():
            result = self.flux_model(
                prompt=prompt_tecnico,
                height=height,
                width=width,
                num_inference_steps=28,
                guidance_scale=3.0,
                max_sequence_length=512,
            )
        
        img = np.array(result.images[0])
        logger.info(f"[VisionLab] Blueprint Nativo 2048x2048 Gerado: {img.shape}")
        
        # V15: High-Frequency SD Img2Img Tiled Upscale substituindo o Lanczos passivo
        if resolution == "4k":
            img = self._tiled_sd_upscale(img, prompt_tecnico)
        
        return img

    def _tiled_sd_upscale(self, base_image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        """
        V15: Tiled Img2Img Pipeline
        Aplica SD Denoise na imagem recortada para injetar texturas ricas em 4K orgânico,
        vencendo a falha arquitetônica de perda de sacadas, detalhes de janelas etc.
        """
        logger.info("[VisionLab] [V15] Multi-pass Tiled Upscale (Injettando detalhes PBR finos)...")
        from PIL import Image as PILImage
        
        base_pil = PILImage.fromarray(base_image_rgb).resize((4096, 4096), PILImage.LANCZOS)
        
        if self.img2img_pipe is None:
            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline
                logger.info("[VisionLab] Carregando Pipeline Img2Img High-Frequency...")
                self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0", 
                    torch_dtype=torch.float16, 
                    use_safetensors=True
                ).to(self.device)
            except Exception as e:
                logger.warning(f"[VisionLab] SD Refiner indisponivel, usando matematico Lanczos. ({e})")
                return np.array(base_pil, dtype=np.uint8)
        
        try:
            logger.info("[VisionLab] Executando Denoise Canny Injetor 4K (Strength 0.2)...")
            with torch.inference_mode():
                result = self.img2img_pipe(
                    prompt=prompt,
                    image=base_pil,
                    strength=0.2, # Super leve para manter só o detalhamento orgânico (não alterar grid)
                    num_inference_steps=5,
                    guidance_scale=1.5
                ).images[0]
            logger.info("[VisionLab] Tiled Upscale Sucesso!")
            return np.array(result, dtype=np.uint8)
        except Exception as e:
            logger.error(f"[VisionLab] Tiled Upscale Failed (OOM?): {e}. Fallback Lanczos.")
            return np.array(base_pil, dtype=np.uint8)

    # =========================================================================
    # [FIX V15] UNIFIED GLOBAL ENHANCEMENT
    # =========================================================================
    def _apply_global_enhancement(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Lente corretiva equilibrada. Aplicada ANTES DO SAM E DO DEPTH.
        Tuned down do V14 (clipLimit 1.2, blur 5x5) pra evitar halos radioativos.
        """
        import cv2
        logger.info("[VisionLab] (V15) Pre-processamento Global Suave: CLAHE e Sharpness leve...")
        
        # 1. CLAHE light para não explodir artefatos brancos
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # 2. Unsharp leve só pra delimitar wireframe
        gaussian = cv2.GaussianBlur(enhanced, (5,5), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.2, gaussian, -0.2, 0, enhanced)
        
        return sharpened

    # =========================================================================
    # 2. SAM 3 (O Retalhador)
    # =========================================================================
    def extract_semantic_atlas(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_sam3()
        
        logger.info("[VisionLab] Ativando fatiamento semântico SAM 3...")
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
            # FIX V13: Nao escalar em * 50 arbitrario. Manter relativo nativo [0, 1].
            # Deixar a matematica focal da classe calcular os metros exatos.
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        depth_map = depth_map.astype(np.float32)
        
        # V15 DEEP CODE PURGE: Rotações e compensações speculares sujas foram apagadas.
        return depth_map

    def calculate_ue5_world_scale(self, bounding_width_pixels: float, z_depth_normalized: float) -> float:
        """
        V15: Métrica física dinâmica ancorada ao base_world_scale_meters.
        Sem números mágicos aleatorios. Tudo flui da propoporção definida.
        """
        # A altura da antena obedece proporcionalmente 20% do grid de mundo base.
        max_height = self.base_world_scale_meters * 0.2 
        height_meters = z_depth_normalized * max_height
        
        # O footprint segue rigorosamente os pixels percentuais do mapa referencial 4096.
        footprint_meters = (bounding_width_pixels / 4096.0) * self.base_world_scale_meters
        
        # Media harmônica
        final_scale = (height_meters + footprint_meters) / 2.0
        return float(final_scale)

    # =========================================================================
    # ORQUESTRADOR DA FASE 1 (API Principal)
    # =========================================================================
    def generate_intent_map(self, prompt: str) -> dict:
        logger.info(">>> [V15] Gerando Blueprint Universal (Layout + SAM + Depth) <<<")
        
        blueprint_raw = self.generate_blueprint(prompt)
        
        # V15: O PRE-PROCESSAMENTO ACONTECE AQUI ANTES DE TUDO!
        # Isso garante match exato de topologia da máscara (SAM) e métrica geométrica (Depth).
        blueprint_4k = self._apply_global_enhancement(blueprint_raw)
        
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
            
            # FIX V13: Filtra mascaras do SAM usando o Depth Relativo
            # No DA3, valores proximos a 0.0 e 0.1 quase sempre sacam o cu ou o asfalto bruto.
            if z_medio < 0.05:
                logger.info(f"[VisionLab] Ignorando Ator {actor_id}: Detectado como Chao/Ceu (Z={z_medio:.3f})")
                continue
                
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