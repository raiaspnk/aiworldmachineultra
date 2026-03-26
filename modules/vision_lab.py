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
# AWE V17-Lite — VisionLab (Lean & Day 1 Ready)
# =============================================================================
# Pipeline simples e testável:
#   1. FLUX.2 single-pass → 2048 nativo → Lanczos+Sharpen 4096
#   2. Enhancement Global Adaptativo (CLAHE + Unsharp)
#   3. SAM 3 → Semantic Atlas
#   4. Depth Anything V2 → Normalized [0, 1]
#   5. Metadata → Diagonal-based UE5 Scale
# =============================================================================

sys.path.append(os.path.abspath("sam3"))


class VisionLab:
    """V17-Lite: Lean single-pass pipeline. Zero overengineering."""

    def __init__(self, base_world_scale_meters: float = 500.0, device: str = "cuda",
                 seed: int = 42):
        self.base_world_scale_meters = base_world_scale_meters
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed

        self.flux_model = None
        self.sam_model = None
        self.sam_generator = None
        self.depth_model = None
        self.depth_processor = None

        logger.info(f"[VisionLab V17-Lite] Scale={base_world_scale_meters}m | Seed={seed} | Device={self.device}")

    # =========================================================================
    # MODEL LOADERS (Lazy)
    # =========================================================================
    def _load_flux(self):
        if self.flux_model is not None:
            return
        logger.info("[VisionLab] Carregando FLUX.2-dev...")
        try:
            from diffusers import FluxPipeline

            self.flux_model = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-dev",
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(self.device)

            if hasattr(self.flux_model, "tokenizer") and self.flux_model.tokenizer is not None:
                if not hasattr(self.flux_model.tokenizer, 'model_max_length'):
                    self.flux_model.tokenizer.model_max_length = 512
            if hasattr(self.flux_model, 'enable_attention_slicing'):
                self.flux_model.enable_attention_slicing()
            logger.info("[VisionLab] FLUX.2-dev operacional.")
        except ImportError:
            logger.error("[VisionLab] 'diffusers' não instalado!")
            raise

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
    # 1. BLUEPRINT (FLUX single-pass → Lanczos+Sharpen 4096)
    # =========================================================================
    def generate_blueprint(self, user_prompt: str, resolution: str = "4k") -> np.ndarray:
        self._load_flux()

        prompt = (
            f"{user_prompt}, strict orthogonal top-down projection, "
            "clear separation between distinct objects and terrain, "
            "well-defined volumetric silhouettes, sharp geometric edges, "
            "high contrast object boundaries, professional overhead scan, "
            "8k resolution"
        )
        logger.info(f"[VisionLab] Prompt: {prompt[:100]}...")

        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        with torch.inference_mode():
            result = self.flux_model(
                prompt=prompt,
                height=2048,
                width=2048,
                num_inference_steps=28,
                guidance_scale=3.5,
                max_sequence_length=512,
                generator=generator,
            )

        img = np.array(result.images[0])
        logger.info(f"[VisionLab] Blueprint nativo: {img.shape}")

        if resolution == "4k":
            import cv2
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img).resize((4096, 4096), PILImage.LANCZOS)
            upscaled = np.array(pil_img, dtype=np.uint8)
            gaussian = cv2.GaussianBlur(upscaled, (3, 3), 1.0)
            img = cv2.addWeighted(upscaled, 1.3, gaussian, -0.3, 0)
            logger.info("[VisionLab] Upscale 4096 Lanczos+Sharpen concluído.")

        return img

    # =========================================================================
    # 2. ENHANCEMENT GLOBAL ADAPTATIVO (antes de SAM E Depth)
    # =========================================================================
    def _apply_global_enhancement(self, image_rgb: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edge_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if edge_var > 150:
            clip, weight = 3.0, 1.4
        elif edge_var > 50:
            clip, weight = 2.0, 1.3
        else:
            clip, weight = 1.2, 1.15

        logger.info(f"[VisionLab] Enhancement: edge_var={edge_var:.0f} → CLAHE={clip}")

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        gaussian = cv2.GaussianBlur(enhanced, (5, 5), 3.0)
        return cv2.addWeighted(enhanced, weight, gaussian, -(weight - 1.0), 0)

    # =========================================================================
    # 3. SAM 3 (Fatiamento Semântico)
    # =========================================================================
    def extract_semantic_atlas(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_sam3()
        height, width = image_rgb.shape[:2]

        if self.sam_model == "FALLBACK_OPENCV":
            return self._sam_fallback_opencv(image_rgb)

        masks = self.sam_generator.generate(image_rgb)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        semantic_map = np.full((height, width), -1, dtype=np.int32)
        for aid, m in enumerate(masks, start=1):
            mb = m['segmentation']
            unfilled = semantic_map[mb] == -1
            idx = np.where(mb)
            semantic_map[idx[0][unfilled], idx[1][unfilled]] = aid

        logger.info(f"[VisionLab] SAM 3: {len(masks)} atores.")
        return semantic_map

    def _sam_fallback_opencv(self, image_rgb: np.ndarray) -> np.ndarray:
        import cv2
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sem = np.full((h, w), -1, dtype=np.int32)
        aid = 1
        for c in contours:
            if cv2.contourArea(c) > 500:
                cv2.drawContours(sem, [c], -1, aid, cv2.FILLED)
                aid += 1
        return sem

    # =========================================================================
    # 4. DEPTH (Normalizada [0, 1])
    # =========================================================================
    def extract_metric_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_depth()

        inputs = self.depth_processor(
            images=Image.fromarray(image_rgb), return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            pred = self.depth_model(**inputs).predicted_depth

        h, w = image_rgb.shape[:2]
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            depth = (depth - d_min) / (d_max - d_min)

        return depth.astype(np.float32)

    # =========================================================================
    # 5. SCALE (Diagonal-based, zero magic numbers)
    # =========================================================================
    def calculate_ue5_world_scale(self, bbox_w: float, bbox_h: float,
                                   z_norm: float, image_size: float) -> float:
        diag = math.sqrt(bbox_w ** 2 + bbox_h ** 2)
        footprint_frac = diag / (image_size * math.sqrt(2))
        xy = footprint_frac * self.base_world_scale_meters
        z = max(z_norm, 0.01) * self.base_world_scale_meters * footprint_frac
        return float(math.sqrt(xy * z))

    # =========================================================================
    # ORQUESTRADOR (API Principal)
    # =========================================================================
    def generate_intent_map(self, prompt: str) -> dict:
        logger.info(">>> [V17-Lite] Intent Map <<<")

        blueprint_raw = self.generate_blueprint(prompt)
        blueprint = self._apply_global_enhancement(blueprint_raw)

        sam_mask = self.extract_semantic_atlas(blueprint)
        depth_map = self.extract_metric_depth(blueprint)

        unique = np.unique(sam_mask)
        unique = unique[unique > 0]
        img_size = float(blueprint.shape[1])
        actors = {}

        for aid in unique:
            mask = (sam_mask == aid)
            rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            z = float(np.mean(depth_map[mask]))

            if z < 0.05:
                continue

            actors[int(aid)] = {
                "ue5_scale_meters": self.calculate_ue5_world_scale(
                    float(cmax - cmin), float(rmax - rmin), z, img_size),
                "z_socket_depth": z,
                "bbox": [int(rmin), int(cmin), int(rmax), int(cmax)],
                "pixel_area": int(np.sum(mask)),
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[VisionLab] {len(actors)} atores válidos.")
        return {
            "blueprint_rgb": blueprint,
            "sam_mask": sam_mask,
            "depth_map": depth_map,
            "actors_metadata": actors,
        }

    # =========================================================================
    # VRAM MANAGEMENT
    # =========================================================================
    def unload_all(self):
        logger.info("[VisionLab] Expurgando VRAM...")
        for attr in ['flux_model', 'sam_model', 'depth_model']:
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self.sam_generator = None
        self.depth_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("[VisionLab] VRAM limpa.")