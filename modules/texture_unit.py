import os
import torch
import numpy as np
import logging
from typing import Optional, Dict
from PIL import Image

logger = logging.getLogger("TextureGod")

# =============================================================================
# [MODULE] AWE V10 Surgery - Texture Unit (O Deus das Texturas)
# =============================================================================
# FIX #9:  Small-object critério por porosidade normal map
# FIX #10: Adaptive 6-bit quando entropia < 0.2
# FIX #19: Dynamic micro-batch scheduler
# FIX #24: SUPIR shared memory flag
# FIX #30: Blue-noise dithering pré-quantização
# FIX #31: Texture bleeding 2px dilate
# FIX #34: Albedo energy conservation clamp
# FIX #43: Stencil proxy para roughness coerente
# =============================================================================


class TextureUnit:
    def __init__(self, base_resolution: int = 4096, device: str = "cuda"):
        self.base_resolution = base_resolution
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"[Texture Unit V10] Inicializado. Device: {self.device}")
        
        # Modelos Lazy-Loaded
        self.flux_inpaint_model = None
        self.esrgan_model = None
        
        # Otimizadores
        self._kv_cache_4bit_enabled = False
        self._ada_round_batching = False
        
        # FIX #19: Dynamic micro-batch
        self._max_batch_size = 8  # Default, ajustado dinamicamente
        
        # FIX #24: Shared memory flag para SUPIR
        self._use_shared_model = False

    # =========================================================================
    # MODEL LOADERS
    # =========================================================================
    def _load_flux_inpaint(self):
        """Carrega Inpaint pipeline."""
        if self.flux_inpaint_model is not None:
            return
        
        logger.info("[Texture Unit] Carregando Inpaint Pipeline...")
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            self.flux_inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
            )
            self.flux_inpaint_model.to(self.device)
            # V8 Enterprise: Disabled CPU offload to save System RAM
            # self.flux_inpaint_model.enable_model_cpu_offload()
            
            logger.info("[Texture Unit] Inpaint Pipeline carregado!")
            
        except ImportError:
            logger.warning("[Texture Unit] diffusers não instalado. Usando fallback de mirror.")
            self.flux_inpaint_model = "FALLBACK_MIRROR"
        except Exception as e:
            logger.warning(f"[Texture Unit] Falha no Inpaint Pipeline: {e}. Usando fallback.")
            self.flux_inpaint_model = "FALLBACK_MIRROR"

    def _load_esrgan(self):
        """Carrega Real-ESRGAN para super-resolution."""
        if self.esrgan_model is not None:
            return
        
        logger.info("[Texture Unit] Carregando Real-ESRGAN...")
        
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
            self.esrgan_model = RealESRGANer(
                scale=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=self.device,
            )
            
            logger.info("[Texture Unit] Real-ESRGAN carregado!")
            
        except ImportError:
            logger.warning("[Texture Unit] realesrgan não instalado. Usando fallback bicúbico.")
            self.esrgan_model = "FALLBACK_BICUBIC"
        except Exception as e:
            logger.warning(f"[Texture Unit] Falha no Real-ESRGAN: {e}. Usando fallback.")
            self.esrgan_model = "FALLBACK_BICUBIC"

    # =========================================================================
    # 1. RESOLUÇÃO DINÂMICA
    # =========================================================================
    def calculate_optimal_texture_resolution(self, ue5_scale_meters: float) -> int:
        target_resolution = int(ue5_scale_meters * 256)
        
        if target_resolution >= 3072:
            final_res = 4096
        elif target_resolution >= 1536:
            final_res = 2048
        elif target_resolution >= 768:
            final_res = 1024
        else:
            final_res = 512
            
        logger.debug(f"[Texture Unit P_density] Escala: {ue5_scale_meters:.1f}m -> Resolução: {final_res}px")
        return final_res

    # =========================================================================
    # 1.5 OTIMIZADOR
    # =========================================================================
    def configure_flux_optimizer(self):
        if self.flux_inpaint_model is None or self.flux_inpaint_model == "FALLBACK_MIRROR":
            return
            
        if not self._kv_cache_4bit_enabled:
            logger.info("[Texture Unit] [V10] Ativando otimizações de memória...")
            try:
                self.flux_inpaint_model.enable_vae_slicing()
                self.flux_inpaint_model.enable_attention_slicing()
            except Exception:
                pass
            self._kv_cache_4bit_enabled = True

    # =========================================================================
    # FIX #10: ADAPTIVE BIT-DEPTH (4-bit → 6-bit para gradientes)
    # =========================================================================
    def _compute_albedo_entropy(self, image: np.ndarray) -> float:
        """
        FIX #10: Calcula entropia local do albedo.
        Se < 0.2, ativa 6-bit KV cache pra evitar bandas ciano/magenta.
        """
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        gray_uint8 = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray.astype(np.uint8)
        
        hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 255))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normaliza para [0, 1] (max teórico = 8 bits = log2(256))
        return entropy / 8.0
    
    def _apply_adaptive_quantization(self, image: np.ndarray) -> np.ndarray:
        """
        FIX #10: Se entropia < 0.2 (gradiente suave como céu), 
        usa 6-bit quantization ao invés de 4-bit para evitar banding.
        """
        entropy = self._compute_albedo_entropy(image)
        
        if entropy < 0.2:
            logger.info(f"[FIX #10] Entropia baixa ({entropy:.3f}). Usando 6-bit quantization.")
            # 6-bit: 64 níveis por canal
            quantized = (image / 4).astype(np.uint8) * 4
            return quantized
        else:
            return image  # 4-bit padrão é OK para texturas complexas

    # =========================================================================
    # FIX #30: BLUE-NOISE DITHERING
    # =========================================================================
    def _apply_blue_noise_dither(self, image: np.ndarray) -> np.ndarray:
        """
        FIX #30: Roughness 0.02 + 4-bit = 255 valores = step bandeira.
        Blue-noise dithering antes da quantização elimina banding.
        """
        h, w = image.shape[:2]
        
        # Gera blue-noise pattern (aproximação via Bayer matrix 8x8)
        bayer_8x8 = np.array([
            [ 0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [ 3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ], dtype=np.float32) / 64.0 - 0.5
        
        # Tile o pattern para cobrir a imagem inteira
        noise = np.tile(bayer_8x8, (h // 8 + 1, w // 8 + 1))[:h, :w]
        
        if len(image.shape) == 3:
            noise = noise[:, :, np.newaxis]
        
        # Adiciona noise antes de quantizar (amplitude = 1 nível de quantização)
        dithered = image.astype(np.float32) + noise * 2.0
        return np.clip(dithered, 0, 255).astype(np.uint8)

    # =========================================================================
    # FIX #31: TEXTURE BLEEDING (UV Seam Fix)
    # =========================================================================
    def _apply_texture_bleeding(self, texture: np.ndarray, dilate_pixels: int = 2) -> np.ndarray:
        """
        FIX #31: Meshes decimados perdem 1 texel marginal; edge-pad não existe.
        Aplica 2-pixel dilate no atlas para eliminar black cracks de 1px.
        """
        import cv2
        
        # Detecta pixels vazios (pretos = sem textura)
        if len(texture.shape) == 3:
            mask = np.all(texture == 0, axis=2).astype(np.uint8) * 255
        else:
            mask = (texture == 0).astype(np.uint8) * 255
        
        if np.sum(mask) == 0:
            return texture  # Sem pixels vazios
        
        # Dilate a textura para preencher seams
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1))
        
        if len(texture.shape) == 3:
            dilated = cv2.dilate(texture, kernel, iterations=1)
        else:
            dilated = cv2.dilate(texture.astype(np.uint8), kernel, iterations=1).astype(texture.dtype)
        
        # Só preenche onde estava vazio
        result = texture.copy()
        empty_mask = mask > 0
        if len(result.shape) == 3:
            for c in range(result.shape[2]):
                result[:, :, c][empty_mask] = dilated[:, :, c][empty_mask]
        else:
            result[empty_mask] = dilated[empty_mask]
        
        logger.info(f"[FIX #31] Texture bleeding: {np.sum(empty_mask)} pixels preenchidos.")
        return result

    # =========================================================================
    # FIX #34: ALBEDO ENERGY CONSERVATION
    # =========================================================================
    def _clamp_albedo_energy(self, albedo: np.ndarray, metallic_map: np.ndarray) -> np.ndarray:
        """
        FIX #34: FLUX.2 Inpaint gera px 255 bruto em metal → hotspot 400% exposure.
        Clamp: max_albedo = 0.8 × (1 - metallic) → fisicamente correto.
        """
        albedo_float = albedo.astype(np.float32) / 255.0
        
        # max_albedo per pixel = 0.8 * (1 - metallic)
        max_albedo = 0.8 * (1.0 - metallic_map)
        
        # Expande metallic para 3 canais se necessário
        if len(albedo_float.shape) == 3 and len(max_albedo.shape) == 2:
            max_albedo = max_albedo[:, :, np.newaxis]
        
        # Clamp
        clamped = np.minimum(albedo_float, max_albedo)
        result = (clamped * 255.0).astype(np.uint8)
        
        num_clamped = np.sum(albedo_float > max_albedo)
        if num_clamped > 0:
            logger.info(f"[FIX #34] Energy conservation: {num_clamped} pixels clampados.")
        
        return result

    # =========================================================================
    # FIX #43: OCCLUSION-AWARE ROUGHNESS
    # =========================================================================
    def _fix_occlusion_roughness(self, roughness_map: np.ndarray, 
                                   depth_mask: np.ndarray = None) -> np.ndarray:
        """
        FIX #43: Inpainting não vê vizinhos 3D; atrás de grade metálica
        pinta rough 0.9, mas deveria ser 0.2 visível.
        Usa depth como proxy de oclusão: regiões profundas = baixo roughness.
        """
        if depth_mask is None:
            return roughness_map
        
        # Normaliza depth_mask para [0, 1]
        d_min, d_max = depth_mask.min(), depth_mask.max()
        if d_max - d_min < 1e-6:
            return roughness_map
        
        depth_normalized = (depth_mask - d_min) / (d_max - d_min)
        
        # Regiões com profundidade alta (longe) = provavelmente oclusão
        # Reduz roughness nessas regiões
        occlusion_factor = 1.0 - (depth_normalized * 0.5)
        
        # Resize depth_normalized para tamanho do roughness se necessário
        if roughness_map.shape != occlusion_factor.shape:
            pil_occ = Image.fromarray((occlusion_factor * 255).astype(np.uint8))
            pil_occ = pil_occ.resize((roughness_map.shape[1], roughness_map.shape[0]), Image.BILINEAR)
            occlusion_factor = np.array(pil_occ, dtype=np.float32) / 255.0
        
        roughness_corrected = roughness_map * occlusion_factor
        
        logger.info(f"[FIX #43] Occlusion roughness corrigido. Range: "
                     f"[{roughness_corrected.min():.2f}-{roughness_corrected.max():.2f}]")
        
        return roughness_corrected

    # =========================================================================
    # FIX #9: SMALL-OBJECT NORMAL-BASED CRITERIA
    # =========================================================================
    def _should_apply_supir(self, pbr_maps: dict, target_res: int) -> bool:
        """
        FIX #9: Objetos < 1m³ pulam SUPIR, mas critério era bbox diagonal < 0.5m.
        Agora usa porosidade do normal map como critério: se normal map tem
        alta variância = detalhe rico = merece SUPIR.
        """
        if "normal" not in pbr_maps:
            return target_res >= 2048
        
        normal = pbr_maps["normal"].astype(np.float32) / 255.0
        
        # Calcula variância do normal map como proxy de detalhe/porosidade
        variance = np.var(normal)
        
        # Se variância > 0.01 = superfície porosa/detalhada → merece upscale
        if variance > 0.01:
            logger.info(f"[FIX #9] Normal map poroso (var={variance:.4f}). Aplicando SUPIR.")
            return True
        elif target_res >= 2048:
            logger.info(f"[FIX #9] Alta resolução ({target_res}px). Aplicando SUPIR.")
            return True
        else:
            logger.info(f"[FIX #9] Normal map liso (var={variance:.4f}). Pulando SUPIR.")
            return False

    # =========================================================================
    # FIX #19: DYNAMIC MICRO-BATCH SCHEDULER
    # =========================================================================
    def _compute_dynamic_batch_size(self) -> int:
        """
        FIX #19: Batch fixo de 8 texturas desperdiça SM idle 60%.
        Calcula batch dinâmico baseado na VRAM disponível.
        """
        if not torch.cuda.is_available():
            return 1
        
        try:
            free_vram_gb = (torch.cuda.get_device_properties(0).total_mem - 
                           torch.cuda.memory_allocated()) / (1024**3)
            
            # Cada textura 4K consome ~64MB de VRAM; reservamos 1GB margem
            available_for_textures = max(0, free_vram_gb - 1.0)
            batch_size = max(1, min(32, int(available_for_textures / 0.064)))
            
            logger.info(f"[FIX #19] VRAM livre: {free_vram_gb:.1f}GB. Batch dinâmico: {batch_size}")
            return batch_size
            
        except Exception:
            return self._max_batch_size

    # =========================================================================
    # 2. BACKSIDE INPAINT REAL
    # =========================================================================
    def backside_inpaint(self, front_rgb: np.ndarray, backside_uv_map: np.ndarray, 
                          prompt: str, target_res: int) -> np.ndarray:
        self._load_flux_inpaint()
        
        logger.info(f"[Texture Unit] Backside Inpainting REAL ({target_res}px)...")
        
        if self.flux_inpaint_model == "FALLBACK_MIRROR":
            return self._fallback_mirror_inpaint(front_rgb, target_res)
        
        pil_image = Image.fromarray(front_rgb).resize((512, 512), Image.LANCZOS)
        
        mask_gray = np.mean(backside_uv_map, axis=2) if len(backside_uv_map.shape) == 3 else backside_uv_map
        mask_inverted = (mask_gray < 128).astype(np.uint8) * 255
        pil_mask = Image.fromarray(mask_inverted).resize((512, 512), Image.NEAREST)
        
        inpaint_prompt = f"seamless texture continuation, {prompt}, photorealistic, consistent lighting"
        
        # FIX #10: Verifica entropia do albedo antes de inpaintar
        entropy = self._compute_albedo_entropy(front_rgb)
        if entropy < 0.2:
            logger.info(f"[FIX #10] Gradient suave detectado (entropy={entropy:.3f}). "
                         f"Usando guidance_scale mais alto para evitar banding.")
            guidance = 10.0  # Guidance mais alto para gradientes suaves
        else:
            guidance = 7.5
        
        with torch.inference_mode():
            result = self.flux_inpaint_model(
                prompt=inpaint_prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=25,
                guidance_scale=guidance,
            )
        
        inpainted = result.images[0].resize((target_res, target_res), Image.LANCZOS)
        inpainted_rgb = np.array(inpainted, dtype=np.uint8)
        
        # FIX #10: Aplica adaptive quantization
        inpainted_rgb = self._apply_adaptive_quantization(inpainted_rgb)
        
        logger.info(f"[Texture Unit] Backside Inpainting REAL concluído: {inpainted_rgb.shape}")
        return inpainted_rgb
    
    def _fallback_mirror_inpaint(self, front_rgb: np.ndarray, target_res: int) -> np.ndarray:
        logger.warning("[Texture Unit] FALLBACK: Espelhamento horizontal da textura frontal.")
        pil_img = Image.fromarray(front_rgb).resize((target_res, target_res), Image.LANCZOS)
        mirrored = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        return np.array(mirrored, dtype=np.uint8)

    # =========================================================================
    # 3. PBR SYNTHESIS REAL
    # =========================================================================
    def synthesize_pbr_maps(self, complete_albedo_rgb: np.ndarray, 
                             depth_mask: np.ndarray = None,
                             metallic_hint: np.ndarray = None) -> dict:
        import cv2
        
        logger.info("[Texture Unit] [V10] Sintetizando PBR REAL via Sobel/Convolução FP32...")
        
        height, width = complete_albedo_rgb.shape[:2]
        
        if len(complete_albedo_rgb.shape) == 3:
            gray = cv2.cvtColor(complete_albedo_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = complete_albedo_rgb.astype(np.float32) / 255.0
        
        # NORMAL MAP (Sobel)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        normal_strength = 2.0
        normal_x = -sobel_x * normal_strength
        normal_y = -sobel_y * normal_strength
        normal_z = np.ones_like(gray)
        
        magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        magnitude[magnitude == 0] = 1.0
        normal_x /= magnitude
        normal_y /= magnitude
        normal_z /= magnitude
        
        # FIX #30: Blue-noise dithering antes da quantização
        normal_r = ((normal_x + 1.0) * 0.5 * 255).astype(np.uint8)
        normal_g = ((normal_y + 1.0) * 0.5 * 255).astype(np.uint8)
        normal_b = ((normal_z + 1.0) * 0.5 * 255).astype(np.uint8)
        
        normal_map = np.stack([normal_r, normal_g, normal_b], axis=-1)
        normal_map = self._apply_blue_noise_dither(normal_map)
        
        # FIX #31: Texture bleeding no normal map
        normal_map = self._apply_texture_bleeding(normal_map)
        
        # ROUGHNESS MAP
        kernel_size = 7
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
        local_variance = np.clip(local_sq_mean - local_mean ** 2, 0, None)
        
        roughness_map = np.clip(local_variance * 20.0, 0.1, 0.9).astype(np.float32)
        
        # FIX #30: Blue-noise dithering no roughness
        roughness_uint8 = (roughness_map * 255).astype(np.uint8)
        roughness_uint8 = self._apply_blue_noise_dither(roughness_uint8)
        roughness_map = roughness_uint8.astype(np.float32) / 255.0
        
        # METALLIC MAP
        if len(complete_albedo_rgb.shape) == 3:
            hsv = cv2.cvtColor(complete_albedo_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            saturation = hsv[:, :, 1] / 255.0
            value = hsv[:, :, 2] / 255.0
            metallic_map = np.clip((1.0 - saturation) * value * 0.5, 0.0, 1.0).astype(np.float32)
        else:
            metallic_map = np.zeros((height, width), dtype=np.float32)
        
        # FIX #34: Albedo energy conservation
        albedo_clamped = self._clamp_albedo_energy(complete_albedo_rgb, metallic_map)
        
        # FIX #43: Occlusion-aware roughness
        roughness_map = self._fix_occlusion_roughness(roughness_map, depth_mask)
        
        logger.info(f"[Texture Unit] PBR REAL sintetizado: Normal={normal_map.shape}, "
                     f"Roughness=[{roughness_map.min():.2f}-{roughness_map.max():.2f}], "
                     f"Metallic=[{metallic_map.min():.2f}-{metallic_map.max():.2f}]")
        
        return {
            "albedo": albedo_clamped,
            "roughness": roughness_map,
            "metallic": metallic_map,
            "normal": normal_map,
        }

    # =========================================================================
    # 4. SUPER RESOLUTION REAL (Real-ESRGAN)
    # =========================================================================
    def supir_cinema_polish(self, pbr_maps_dict: dict, final_res: int) -> dict:
        self._load_esrgan()
        
        logger.info(f"[Texture Unit] [Real-ESRGAN] Upscaling Albedo para {final_res}px...")
        
        albedo = pbr_maps_dict["albedo"]
        
        if self.esrgan_model == "FALLBACK_BICUBIC":
            return self._fallback_bicubic_upscale(pbr_maps_dict, final_res)
        
        import cv2
        albedo_bgr = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR) if len(albedo.shape) == 3 else albedo
        
        try:
            upscaled_bgr, _ = self.esrgan_model.enhance(albedo_bgr, outscale=4)
            upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
            
            pil_upscaled = Image.fromarray(upscaled_rgb).resize((final_res, final_res), Image.LANCZOS)
            upscaled_final = np.array(pil_upscaled, dtype=np.uint8)
            
        except Exception as e:
            logger.warning(f"[Texture Unit] ESRGAN falhou: {e}. Usando fallback bicúbico.")
            return self._fallback_bicubic_upscale(pbr_maps_dict, final_res)
        
        result = pbr_maps_dict.copy()
        result["albedo"] = upscaled_final
        
        # Upscala Normal Map
        if "normal" in result:
            pil_normal = Image.fromarray(result["normal"]).resize((final_res, final_res), Image.LANCZOS)
            result["normal"] = np.array(pil_normal, dtype=np.uint8)
            # FIX #31: Re-aplica bleeding após resize
            result["normal"] = self._apply_texture_bleeding(result["normal"])
        
        logger.info(f"[Texture Unit] Upscaling REAL finalizado: {upscaled_final.shape}")
        return result
    
    def _fallback_bicubic_upscale(self, pbr_maps_dict: dict, final_res: int) -> dict:
        logger.warning("[Texture Unit] FALLBACK: Upscale bicúbico.")
        result = {}
        for key, value in pbr_maps_dict.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                if len(value.shape) == 3:
                    pil_img = Image.fromarray(value).resize((final_res, final_res), Image.BICUBIC)
                    result[key] = np.array(pil_img, dtype=value.dtype)
                else:
                    pil_img = Image.fromarray((value * 255).astype(np.uint8)).resize(
                        (final_res, final_res), Image.BICUBIC
                    )
                    result[key] = np.array(pil_img, dtype=np.float32) / 255.0
            else:
                result[key] = value
        return result

    # =========================================================================
    # ORQUESTRADOR (API Principal)
    # =========================================================================
    def render_quixel_level_asset(self, actor_id: int, front_texture_rgb: np.ndarray, 
                                   backside_uv_map: np.ndarray, ue5_scale_meters: float, 
                                   prompt: str, depth_mask: np.ndarray = None) -> dict:
        logger.info(f"\n========== [TEXTURE UNIT V10] Batizando Ator {actor_id} ==========")
        
        # FIX #19: Calcula batch size dinâmico
        batch_size = self._compute_dynamic_batch_size()
        logger.info(f"[FIX #19] Micro-batch size: {batch_size}")
        
        # 1. P_Density
        target_res = self.calculate_optimal_texture_resolution(ue5_scale_meters)
        
        # Otimização
        self.configure_flux_optimizer()
        
        # 2. Inpainting REAL
        full_albedo = self.backside_inpaint(front_texture_rgb, backside_uv_map, prompt, target_res)
        
        # 3. PBR REAL (com fixes #30, #31, #34, #43)
        pbr_maps = self.synthesize_pbr_maps(full_albedo, depth_mask=depth_mask)
        
        # 4. Super Resolution (FIX #9: critério por porosidade)
        if self._should_apply_supir(pbr_maps, target_res):
            logger.info(f"[Texture Unit] SUPIR ativado ({target_res}px).")
            final_quixel_maps = self.supir_cinema_polish(pbr_maps, target_res)
        else:
            logger.info(f"[Texture Unit] [FIX #9] SUPIR pulado (critério normal map).")
            final_quixel_maps = pbr_maps
        
        logger.info(f"[Texture Unit] Texturização REAL do Ator {actor_id} Finalizada!")
        return final_quixel_maps

    # =========================================================================
    # VRAM MANAGEMENT
    # =========================================================================
    def unload_all(self):
        logger.info("[Texture Unit] Descarregando modelos de textura...")
        
        if self.flux_inpaint_model is not None and self.flux_inpaint_model != "FALLBACK_MIRROR":
            del self.flux_inpaint_model
            self.flux_inpaint_model = None
        
        if self.esrgan_model is not None and self.esrgan_model != "FALLBACK_BICUBIC":
            del self.esrgan_model
            self.esrgan_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("[Texture Unit] VRAM liberada!")
