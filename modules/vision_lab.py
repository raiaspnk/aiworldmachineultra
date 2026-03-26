import os
import sys
import pkgutil
import gc
import math

# =============================================================================
# Monkey Patch para Python 3.12 (triton/setuptools compatibility)
# =============================================================================
if not hasattr(pkgutil, 'ImpImporter'):
    class DummyImpImporter: pass
    pkgutil.ImpImporter = DummyImpImporter

import torch
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger("VisionLab")

# =============================================================================
# [MODULE] AWE V17 — VisionLab (The Universal Architecture)
# =============================================================================
# Pipeline:
#   1. generate_blueprint  → FLUX Pass 1 (rough) → Canny extract → FLUX Pass 2 (guided)
#   2. _apply_global_enhancement → Adaptive CLAHE + Unsharp (before SAM AND Depth)
#   3. extract_semantic_atlas → SAM 3
#   4. extract_metric_depth  → Depth Anything V2
#   5. generate_intent_map   → Metadata (diagonal-based scale)
# =============================================================================

sys.path.append(os.path.abspath("sam3"))


class VisionLab:
    """V17: Two-Pass ControlNet + True Universal World Engine Vision Pipeline."""

    def __init__(self, base_world_scale_meters: float = 500.0, device: str = "cuda",
                 use_tiled_upscale: bool = False, seed: int = 42):
        self.base_world_scale_meters = base_world_scale_meters
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_tiled_upscale = use_tiled_upscale
        self.seed = seed

        self.flux_pipeline = None       # Always FluxPipeline (text-only for Pass 1)
        self.flux_cn_pipeline = None    # FluxControlNetPipeline (for Pass 2, if available)
        self.controlnet = None
        self.sam_model = None
        self.sam_generator = None
        self.depth_model = None
        self.depth_processor = None
        self.img2img_pipe = None

        logger.info(f"[VisionLab V17] Scale={base_world_scale_meters}m | Seed={seed} | Device={self.device}")

    # =========================================================================
    # MODEL LOADERS (Lazy)
    # =========================================================================
    def _load_flux(self):
        if self.flux_pipeline is not None:
            return
        logger.info("[VisionLab] Carregando FLUX.2...")
        try:
            from diffusers import FluxPipeline

            self.flux_pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(self.device)

            if hasattr(self.flux_pipeline, "tokenizer") and self.flux_pipeline.tokenizer is not None:
                if not hasattr(self.flux_pipeline.tokenizer, 'model_max_length'):
                    self.flux_pipeline.tokenizer.model_max_length = 512
            if hasattr(self.flux_pipeline, 'enable_attention_slicing'):
                self.flux_pipeline.enable_attention_slicing()
            logger.info("[VisionLab] FLUX.2 operacional.")

        except ImportError:
            logger.error("[VisionLab] 'diffusers' não instalado!")
            raise
        except Exception as e:
            logger.error(f"[VisionLab] Falha crítica FLUX: {e}")
            raise

    def _load_controlnet(self):
        """Carrega ControlNet separadamente (só quando necessário pro Pass 2)."""
        if self.flux_cn_pipeline is not None:
            return True
        try:
            from diffusers import FluxControlNetPipeline, FluxControlNetModel

            logger.info("[VisionLab] Acoplando ControlNet-Union pro Pass 2...")
            self.controlnet = FluxControlNetModel.from_pretrained(
                "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
                torch_dtype=torch.bfloat16
            ).to(self.device)
            self.flux_cn_pipeline = FluxControlNetPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                controlnet=self.controlnet,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(self.device)
            if hasattr(self.flux_cn_pipeline, 'enable_attention_slicing'):
                self.flux_cn_pipeline.enable_attention_slicing()
            logger.info("[VisionLab] ControlNet-Union acoplado!")
            return True
        except Exception as e:
            logger.warning(f"[VisionLab] ControlNet indisponível ({e}). Pass 2 será ignorado.")
            return False

    def _load_sam3(self):
        if self.sam_model is not None:
            return
        try:
            from sam3.build_sam import build_sam3
            from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator

            self.sam_model = build_sam3(
                config_file="sam3_hiera_l.yaml",
                ckpt_path="facebook/sam3-hiera-large",
                device=self.device,
            )
            self.sam_generator = SAM3AutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=500,
            )
            logger.info("[VisionLab] SAM 3 operacional.")
        except Exception as e:
            logger.warning(f"[VisionLab] SAM 3 indisponível ({e}). Fallback OpenCV.")
            self.sam_model = "FALLBACK_OPENCV"

    def _load_depth(self):
        if self.depth_model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            self.depth_processor = AutoImageProcessor.from_pretrained(model_id)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to(self.device)
            logger.info("[VisionLab] Depth Anything V2 operacional.")
        except ImportError:
            logger.error("[VisionLab] 'transformers' não instalado!")
            raise

    # =========================================================================
    # 1. BLUEPRINT GENERATION — TWO-PASS CONTROLNET
    # =========================================================================
    def _build_universal_prompt(self, user_prompt: str) -> str:
        """
        V17: Prompt universal que NÃO força layout urbano.
        Condiciona projeção ortogonal e separação de objetos sem impor grid de rua.
        """
        return (
            f"{user_prompt}, strict orthogonal top-down projection, "
            "clear separation between distinct objects and terrain, "
            "well-defined volumetric silhouettes, sharp geometric edges, "
            "high contrast object boundaries, professional overhead scan, "
            "8k resolution"
        )

    def _extract_canny_from_image(self, image_rgb: np.ndarray) -> Image.Image:
        """Extrai bordas Canny REAIS da imagem gerada no Pass 1."""
        import cv2
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Bilateral filter pra suavizar ruído mas manter bordas
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 50, 150)
        # Dilatação leve pra conectar bordas quebradas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(canny_rgb)

    def generate_blueprint(self, user_prompt: str, resolution: str = "4k") -> np.ndarray:
        """
        V17: Two-Pass ControlNet Strategy.
        
        Pass 1: FLUX puro gera blueprint rough (respeitando o conteúdo do prompt).
        Pass 2: Canny extraído do rough → ControlNet guia refinamento com geometria REAL.
        
        Se ControlNet não estiver disponível, retorna o Pass 1 apenas.
        """
        self._load_flux()

        prompt = self._build_universal_prompt(user_prompt)
        width, height = 2048, 2048
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # ── PASS 1: FLUX puro (rough layout baseado no prompt do usuário) ──
        logger.info("[VisionLab] Pass 1: FLUX rough layout...")
        with torch.inference_mode():
            rough_result = self.flux_pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=20,  # Menos steps (é rascunho)
                guidance_scale=3.5,
                max_sequence_length=512,
                generator=generator,
            )
        rough_img = np.array(rough_result.images[0])
        logger.info(f"[VisionLab] Pass 1 concluído: {rough_img.shape}")

        # ── PASS 2: ControlNet guiado pelo Canny REAL do rough ──
        if self._load_controlnet():
            canny_control = self._extract_canny_from_image(rough_img)
            generator_p2 = torch.Generator(device=self.device).manual_seed(self.seed)

            logger.info("[VisionLab] Pass 2: FLUX + ControlNet Canny (geometria real guiada)...")
            with torch.inference_mode():
                refined_result = self.flux_cn_pipeline(
                    prompt=prompt,
                    control_image=canny_control,
                    controlnet_conditioning_scale=0.55,
                    height=height,
                    width=width,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                    max_sequence_length=512,
                    generator=generator_p2,
                )
            img = np.array(refined_result.images[0])
            logger.info(f"[VisionLab] Pass 2 concluído: {img.shape}")
        else:
            logger.warning("[VisionLab] ControlNet indisponível. Usando resultado do Pass 1.")
            img = rough_img

        # ── UPSCALE ──
        if resolution == "4k":
            img = self._tiled_sd_upscale(img, prompt) if self.use_tiled_upscale else self._lanczos_sharpen_upscale(img)

        return img

    # =========================================================================
    # UPSCALE STRATEGIES
    # =========================================================================
    def _lanczos_sharpen_upscale(self, img: np.ndarray) -> np.ndarray:
        """Lanczos 4096 + post-sharpen leve. Default seguro."""
        import cv2
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img).resize((4096, 4096), PILImage.LANCZOS)
        upscaled = np.array(pil_img, dtype=np.uint8)
        gaussian = cv2.GaussianBlur(upscaled, (3, 3), 1.0)
        sharpened = cv2.addWeighted(upscaled, 1.3, gaussian, -0.3, 0)
        logger.info("[VisionLab] Upscale Lanczos+Sharpen 4096 concluído.")
        return sharpened

    def _tiled_sd_upscale(self, base_image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        """Opcional: SD Refiner img2img. Só ativa com use_tiled_upscale=True."""
        from PIL import Image as PILImage

        base_pil = PILImage.fromarray(base_image_rgb).resize((4096, 4096), PILImage.LANCZOS)
        if self.img2img_pipe is None:
            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline
                self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    torch_dtype=torch.float16, use_safetensors=True
                ).to(self.device)
            except Exception as e:
                logger.warning(f"[VisionLab] SD Refiner indisponível ({e}). Fallback Lanczos.")
                return np.array(base_pil, dtype=np.uint8)

        try:
            with torch.inference_mode():
                result = self.img2img_pipe(
                    prompt=prompt, image=base_pil,
                    strength=0.15, num_inference_steps=4, guidance_scale=1.2
                ).images[0]
            return np.array(result, dtype=np.uint8)
        except Exception as e:
            logger.error(f"[VisionLab] SD Upscale OOM: {e}. Fallback Lanczos.")
            return np.array(base_pil, dtype=np.uint8)

    # =========================================================================
    # 2. GLOBAL ENHANCEMENT — Adaptive CLAHE (respects scene content)
    # =========================================================================
    def _apply_global_enhancement(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        V17: CLAHE adaptativo — mede a variância de bordas da imagem.
        Cenas duras (urbano, indústria) → CLAHE forte (3.0).
        Cenas suaves (floresta, água, névoa) → CLAHE leve (1.5).
        """
        import cv2
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edge_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Threshold empírico: cenas urbanas tipicamente >200, orgânicas <100
        if edge_variance > 150:
            clip_limit = 3.0
            unsharp_weight = 1.4
        elif edge_variance > 50:
            clip_limit = 2.0
            unsharp_weight = 1.3
        else:
            clip_limit = 1.2
            unsharp_weight = 1.15

        logger.info(f"[VisionLab] Enhancement Adaptativo: edge_var={edge_variance:.0f} → CLAHE={clip_limit}, unsharp={unsharp_weight}")

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        gaussian = cv2.GaussianBlur(enhanced, (5, 5), 3.0)
        neg_weight = -(unsharp_weight - 1.0)
        sharpened = cv2.addWeighted(enhanced, unsharp_weight, gaussian, neg_weight, 0)
        return sharpened

    # =========================================================================
    # 3. SAM 3 (Fatiamento Semântico)
    # =========================================================================
    def extract_semantic_atlas(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_sam3()
        height, width = image_rgb.shape[:2]

        if self.sam_model == "FALLBACK_OPENCV":
            return self._sam_fallback_opencv(image_rgb)

        masks_output = self.sam_generator.generate(image_rgb)
        masks_output = sorted(masks_output, key=lambda x: x['area'], reverse=True)

        semantic_map = np.full((height, width), -1, dtype=np.int32)
        for actor_id, mask_data in enumerate(masks_output, start=1):
            mask_bool = mask_data['segmentation']
            unfilled = semantic_map[mask_bool] == -1
            idx = np.where(mask_bool)
            semantic_map[idx[0][unfilled], idx[1][unfilled]] = actor_id

        logger.info(f"[VisionLab] SAM 3: {len(masks_output)} atores extraídos.")
        return semantic_map

    def _sam_fallback_opencv(self, image_rgb: np.ndarray) -> np.ndarray:
        """Fallback OpenCV (recebe imagem já enhanced)."""
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
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(semantic_map, [contour], -1, actor_id, cv2.FILLED)
                actor_id += 1
        return semantic_map

    # =========================================================================
    # 4. DEPTH ANYTHING (Normalizada [0, 1])
    # =========================================================================
    def extract_metric_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_depth()

        pil_image = Image.fromarray(image_rgb)
        inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            predicted_depth = self.depth_model(**inputs).predicted_depth

        height, width = image_rgb.shape[:2]
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 0:
            depth_map = (depth_map - d_min) / (d_max - d_min)

        logger.info(f"[VisionLab] Depth map: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        return depth_map.astype(np.float32)

    # =========================================================================
    # 5. SCALE CALCULATION — Diagonal + Depth Fraction (Zero Magic Numbers)
    # =========================================================================
    def calculate_ue5_world_scale(self, bbox_width_px: float, bbox_height_px: float,
                                   z_depth_normalized: float, image_size: float) -> float:
        """
        V17: Escala puramente proporcional derivada do base_world_scale_meters.
        
        footprint_fraction = diagonal do bbox / diagonal máxima da imagem
        depth_fraction     = z_depth normalizado (0..1)
        
        XY scale = footprint_fraction × base_world_scale_meters
        Z  scale = depth_fraction × base_world_scale_meters × footprint_fraction
        
        O Z é modulado pelo footprint para que objetos pequenos com depth alto
        não explodam em escala (um poste fino com z=0.8 não vira um prédio).
        """
        diagonal_px = math.sqrt(bbox_width_px ** 2 + bbox_height_px ** 2)
        max_diagonal = image_size * math.sqrt(2)
        footprint_fraction = diagonal_px / max_diagonal

        xy_scale = footprint_fraction * self.base_world_scale_meters
        z_clamped = max(z_depth_normalized, 0.01)
        z_scale = z_clamped * self.base_world_scale_meters * footprint_fraction

        # Média geométrica entre XY e Z
        return float(math.sqrt(xy_scale * z_scale))

    # =========================================================================
    # ORQUESTRADOR DA FASE 1 (API Principal)
    # =========================================================================
    def generate_intent_map(self, prompt: str) -> dict:
        logger.info(">>> [V17] Intent Map Universal <<<")

        blueprint_raw = self.generate_blueprint(prompt)
        blueprint = self._apply_global_enhancement(blueprint_raw)

        sam_mask = self.extract_semantic_atlas(blueprint)
        depth_map = self.extract_metric_depth(blueprint)

        unique_actors = np.unique(sam_mask)
        unique_actors = unique_actors[unique_actors > 0]

        image_size = float(blueprint.shape[1])
        actors_metadata = {}

        for actor_id in unique_actors:
            actor_mask = (sam_mask == actor_id)
            rows = np.any(actor_mask, axis=1)
            cols = np.any(actor_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_w = float(cmax - cmin)
            bbox_h = float(rmax - rmin)
            z_medio = float(np.mean(depth_map[actor_mask]))

            if z_medio < 0.05:
                continue

            scale = self.calculate_ue5_world_scale(bbox_w, bbox_h, z_medio, image_size)
            actors_metadata[int(actor_id)] = {
                "ue5_scale_meters": scale,
                "z_socket_depth": z_medio,
                "bbox": [int(rmin), int(cmin), int(rmax), int(cmax)],
                "pixel_area": int(np.sum(actor_mask)),
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[VisionLab] Intent Map: {len(actors_metadata)} atores válidos.")
        return {
            "blueprint_rgb": blueprint,
            "sam_mask": sam_mask,
            "depth_map": depth_map,
            "actors_metadata": actors_metadata,
        }

    # =========================================================================
    # VRAM MANAGEMENT
    # =========================================================================
    def unload_all(self):
        logger.info("[VisionLab] Expurgando VRAM...")
        for attr in ['flux_pipeline', 'flux_cn_pipeline', 'controlnet',
                      'sam_model', 'depth_model', 'img2img_pipe']:
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        self.sam_generator = None
        self.depth_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("[VisionLab] VRAM limpa.")