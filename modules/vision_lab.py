import os
import sys
import pkgutil
import gc
import math

# =============================================================================
# [HACK SÊNIOR] Monkey Patch para o Python 3.12 (Mata o erro do triton/setuptools)
# =============================================================================
if not hasattr(pkgutil, 'ImpImporter'):
    class DummyImpImporter: pass
    pkgutil.ImpImporter = DummyImpImporter

import torch
import numpy as np
import logging
from typing import Optional, Dict
from PIL import Image

logger = logging.getLogger("VisionLab")

# =============================================================================
# [MODULE] AWE V16 — VisionLab (The Universal Architecture — Final Purification)
# =============================================================================
# Pipeline Linear Limpo:
#   1. Blueprint 2D (FLUX.2 + ControlNet Canny Guiado)
#   2. Enhancement Global (CLAHE + Unsharp → aplicado antes de SAM E Depth)
#   3. Fatiamento Semântico (SAM 3)
#   4. Profundidade Métrica (Depth Anything V2)
#   5. Metadata Extraction (Diagonal-based UE5 Scale)
# =============================================================================

sys.path.append(os.path.abspath("sam3"))


class VisionLab:
    """V16: Universal World Engine Vision Pipeline."""

    def __init__(self, base_world_scale_meters: float = 500.0, device: str = "cuda",
                 use_tiled_upscale: bool = False, seed: int = 42):
        self.base_world_scale_meters = base_world_scale_meters
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_tiled_upscale = use_tiled_upscale
        self.seed = seed

        # Model slots (lazy loaded)
        self.flux_model = None
        self.controlnet = None
        self.sam_model = None
        self.sam_generator = None
        self.depth_model = None
        self.depth_processor = None
        self.img2img_pipe = None

        logger.info(f"[VisionLab V16] Inicializado. Scale={self.base_world_scale_meters}m | Seed={self.seed} | Device={self.device}")

    # =========================================================================
    # MODEL LOADERS (Lazy)
    # =========================================================================
    def _load_flux(self):
        if self.flux_model is not None:
            return
        logger.info("[VisionLab] Carregando FLUX.2 + ControlNet Canny...")
        try:
            from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel

            try:
                self.controlnet = FluxControlNetModel.from_pretrained(
                    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
                    torch_dtype=torch.bfloat16
                ).to(self.device)
                self.flux_model = FluxControlNetPipeline.from_pretrained(
                    "black-forest-labs/FLUX.2-dev",
                    controlnet=self.controlnet,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                ).to(self.device)
                logger.info("[VisionLab] ControlNet-Union acoplado com sucesso!")
            except Exception as e_cn:
                logger.warning(f"[VisionLab] ControlNet indisponível ({e_cn}). Fallback: FLUX puro.")
                self.controlnet = None
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

        except ImportError:
            logger.error("[VisionLab] 'diffusers' não instalado!")
            raise
        except Exception as e:
            logger.error(f"[VisionLab] Falha crítica FLUX: {e}")
            raise

    def _load_sam3(self):
        if self.sam_model is not None:
            return
        logger.info("[VisionLab] Carregando SAM 3...")
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
            logger.info("[VisionLab] SAM 3 operacional!")
        except (ImportError, Exception) as e:
            logger.warning(f"[VisionLab] SAM 3 indisponível ({e}). Fallback OpenCV.")
            self.sam_model = "FALLBACK_OPENCV"

    def _load_depth(self):
        if self.depth_model is not None:
            return
        logger.info("[VisionLab] Carregando Depth Anything V2...")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            self.depth_processor = AutoImageProcessor.from_pretrained(model_id)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to(self.device)
            logger.info("[VisionLab] Depth Anything V2 operacional!")
        except ImportError:
            logger.error("[VisionLab] 'transformers' não instalado!")
            raise

    # =========================================================================
    # 1. BLUEPRINT GENERATION (FLUX + ControlNet Canny Real)
    # =========================================================================
    def _generate_canny_control_image(self, width: int, height: int) -> Image.Image:
        """
        Gera uma imagem de controle Canny sintética (grid ortogonal procedural).
        Isso dá ao ControlNet algo concreto pra seguir: ruas retas, blocos alinhados.
        Funciona pra qualquer bioma (urbano, medieval, sci-fi) porque o grid é genérico.
        """
        import cv2
        canvas = np.zeros((height, width), dtype=np.uint8)

        # Grid principal de ruas (linhas brancas = bordas Canny)
        num_blocks_x = np.random.RandomState(self.seed).randint(4, 8)
        num_blocks_y = np.random.RandomState(self.seed + 1).randint(4, 8)

        for i in range(1, num_blocks_x):
            x = int(i * width / num_blocks_x)
            jitter = np.random.RandomState(self.seed + i).randint(-15, 15)
            cv2.line(canvas, (x + jitter, 0), (x + jitter, height), 255, thickness=3)

        for j in range(1, num_blocks_y):
            y = int(j * height / num_blocks_y)
            jitter = np.random.RandomState(self.seed + 100 + j).randint(-15, 15)
            cv2.line(canvas, (0, y + jitter), (width, y + jitter), 255, thickness=3)

        # Sub-blocos internos (variação de footprint)
        rng = np.random.RandomState(self.seed + 200)
        for _ in range(rng.randint(8, 20)):
            x1, y1 = rng.randint(0, width), rng.randint(0, height)
            w, h = rng.randint(40, 200), rng.randint(40, 200)
            cv2.rectangle(canvas, (x1, y1), (x1 + w, y1 + h), 255, thickness=2)

        # Converte pra RGB (ControlNet espera 3 canais)
        canny_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(canny_rgb)

    def generate_blueprint(self, user_prompt: str, resolution: str = "4k") -> np.ndarray:
        """V16: Gera blueprint com ControlNet Canny guiando a geometria."""
        self._load_flux()

        # Prompt fortíssimo universal
        prompt_tecnico = (
            f"{user_prompt}, strict orthogonal top-down projection, "
            "consistent modular block layout, aligned volumetric structures, "
            "uniform building density, clean separation between objects, "
            "sharp geometric edges, high contrast structural footprints, "
            "professional architectural overhead scan, 8k resolution"
        )
        logger.info(f"[VisionLab] Prompt V16: {prompt_tecnico[:120]}...")

        width, height = 2048, 2048
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        with torch.inference_mode():
            if self.controlnet is not None:
                # PRIORIDADE 1: ControlNet Canny REALMENTE USADO
                control_image = self._generate_canny_control_image(width, height)
                logger.info("[VisionLab] Executando FLUX + ControlNet Canny (Grid Guiado)...")
                result = self.flux_model(
                    prompt=prompt_tecnico,
                    control_image=control_image,
                    controlnet_conditioning_scale=0.6,
                    height=height,
                    width=width,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                    max_sequence_length=512,
                    generator=generator,
                )
            else:
                logger.info("[VisionLab] Executando FLUX Puro (sem ControlNet)...")
                result = self.flux_model(
                    prompt=prompt_tecnico,
                    height=height,
                    width=width,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                    max_sequence_length=512,
                    generator=generator,
                )

        img = np.array(result.images[0])
        logger.info(f"[VisionLab] Blueprint Nativo: {img.shape}")

        # Upscale: Condicional (Tiled SD ou Lanczos+Sharpen)
        if resolution == "4k":
            if self.use_tiled_upscale:
                img = self._tiled_sd_upscale(img, prompt_tecnico)
            else:
                img = self._lanczos_sharpen_upscale(img)

        return img

    # =========================================================================
    # UPSCALE STRATEGIES
    # =========================================================================
    def _lanczos_sharpen_upscale(self, img: np.ndarray) -> np.ndarray:
        """Lanczos 4096 + Unsharp pós-upscale. Leve e seguro."""
        import cv2
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img).resize((4096, 4096), PILImage.LANCZOS)
        upscaled = np.array(pil_img, dtype=np.uint8)

        # Post-sharpen pra recuperar bordas perdidas no resize
        gaussian = cv2.GaussianBlur(upscaled, (3, 3), 1.0)
        sharpened = cv2.addWeighted(upscaled, 1.3, gaussian, -0.3, 0)
        logger.info("[VisionLab] Upscale Lanczos+Sharpen 4096x4096 concluído.")
        return sharpened

    def _tiled_sd_upscale(self, base_image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        """Opcional: SD Refiner img2img. Só carrega se use_tiled_upscale=True."""
        from PIL import Image as PILImage
        logger.info("[VisionLab] Tiled SD Upscale (OPCIONAL — ativado pelo usuário)...")

        base_pil = PILImage.fromarray(base_image_rgb).resize((4096, 4096), PILImage.LANCZOS)

        if self.img2img_pipe is None:
            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline
                self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    torch_dtype=torch.float16,
                    use_safetensors=True
                ).to(self.device)
            except Exception as e:
                logger.warning(f"[VisionLab] SD Refiner indisponível ({e}). Fallback Lanczos.")
                return np.array(base_pil, dtype=np.uint8)

        try:
            with torch.inference_mode():
                result = self.img2img_pipe(
                    prompt=prompt,
                    image=base_pil,
                    strength=0.15,
                    num_inference_steps=4,
                    guidance_scale=1.2
                ).images[0]
            logger.info("[VisionLab] Tiled SD Upscale concluído!")
            return np.array(result, dtype=np.uint8)
        except Exception as e:
            logger.error(f"[VisionLab] SD Upscale OOM: {e}. Fallback Lanczos.")
            return np.array(base_pil, dtype=np.uint8)

    # =========================================================================
    # 2. GLOBAL ENHANCEMENT (CLAHE + Unsharp — antes de SAM E Depth)
    # =========================================================================
    def _apply_global_enhancement(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        V16: Parâmetros calibrados mais agressivos.
        clipLimit=3.0 separa silhuetas fortes. Unsharp cravar bordas pro SAM.
        """
        import cv2
        logger.info("[VisionLab] Enhancement Global V16 (CLAHE 3.0 + Unsharp Agressivo)...")

        # CLAHE agressivo na luminância
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        # Unsharp forte (raio 5, weight 1.4)
        gaussian = cv2.GaussianBlur(enhanced, (5, 5), 3.0)
        sharpened = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)

        return sharpened

    # =========================================================================
    # 3. SAM 3 (Fatiamento Semântico)
    # =========================================================================
    def extract_semantic_atlas(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_sam3()
        logger.info("[VisionLab] SAM 3 — Fatiamento Semântico...")
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

        logger.info(f"[VisionLab] SAM 3 extraiu {len(masks_output)} atores.")
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
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(semantic_map, [contour], -1, actor_id, cv2.FILLED)
                actor_id += 1
        return semantic_map

    # =========================================================================
    # 4. DEPTH ANYTHING (Régua Métrica Normalizada)
    # =========================================================================
    def extract_metric_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load_depth()
        logger.info("[VisionLab] Depth Anything V2 — Topologia Z...")

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

        # Normalização [0, 1]
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map.astype(np.float32)

    # =========================================================================
    # 5. SCALE CALCULATION (Diagonal-Based — Sem Magic Numbers)
    # =========================================================================
    def calculate_ue5_world_scale(self, bbox_width_px: float, bbox_height_px: float,
                                   z_depth_normalized: float, image_size: float = 4096.0) -> float:
        """
        V16: Escala baseada na diagonal real do bounding box + profundidade relativa.
        
        A diagonal do bbox em pixels é convertida pra fração do mapa total.
        O depth normalizado modula a altura Z.
        base_world_scale_meters é o único parâmetro configurável do usuário.
        """
        # Diagonal real do ator em pixels
        diagonal_px = math.sqrt(bbox_width_px ** 2 + bbox_height_px ** 2)

        # Fração que esse ator ocupa do mapa visual completo
        footprint_fraction = diagonal_px / (image_size * math.sqrt(2))

        # Footprint em metros
        footprint_meters = footprint_fraction * self.base_world_scale_meters

        # Altura proporcional ao depth (z=1.0 = 25% do mapa total, escala logarítmica)
        z_clamped = max(z_depth_normalized, 0.01)
        height_meters = z_clamped * self.base_world_scale_meters * 0.25

        # Média geométrica (preserva proporção sem favorecer nenhum eixo)
        scale = math.sqrt(footprint_meters * height_meters)
        return float(scale)

    # =========================================================================
    # ORQUESTRADOR DA FASE 1 (API Principal)
    # =========================================================================
    def generate_intent_map(self, prompt: str) -> dict:
        logger.info(">>> [V16] Gerando Intent Map Universal <<<")

        # Step 1: Blueprint
        blueprint_raw = self.generate_blueprint(prompt)

        # Step 2: Enhancement Global ANTES de SAM e Depth (coerência garantida)
        blueprint_enhanced = self._apply_global_enhancement(blueprint_raw)

        # Step 3: SAM
        sam_mask = self.extract_semantic_atlas(blueprint_enhanced)

        # Step 4: Depth
        depth_map = self.extract_metric_depth(blueprint_enhanced)

        # Step 5: Metadata Extraction
        unique_actors = np.unique(sam_mask)
        unique_actors = unique_actors[unique_actors > 0]

        image_size = float(blueprint_enhanced.shape[1])
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

            # Filtro Z: ignora chão/céu
            if z_medio < 0.05:
                logger.debug(f"[VisionLab] Ignorando Ator {actor_id} (Z={z_medio:.3f} → chão/céu)")
                continue

            scale_ue5 = self.calculate_ue5_world_scale(bbox_w, bbox_h, z_medio, image_size)

            actors_metadata[int(actor_id)] = {
                "ue5_scale_meters": scale_ue5,
                "z_socket_depth": z_medio,
                "bbox": [int(rmin), int(cmin), int(rmax), int(cmax)],
                "pixel_area": int(np.sum(actor_mask)),
            }

        # Flush VRAM intermediária
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "blueprint_rgb": blueprint_enhanced,
            "sam_mask": sam_mask,
            "depth_map": depth_map,
            "actors_metadata": actors_metadata,
        }

    # =========================================================================
    # VRAM MANAGEMENT
    # =========================================================================
    def unload_all(self):
        logger.info("[VisionLab] Expurgando VRAM...")
        for attr in ['flux_model', 'controlnet', 'sam_model', 'depth_model', 'img2img_pipe']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("[VisionLab] VRAM limpa.")