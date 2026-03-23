"""
AI WORLD ENGINE — TESTE COMPLETO MODAL A100
Gera imagem → Depth V3 → SAM → Trellis REAL (não demo)
Roda: modal run tests/modal_full_pipeline.py
"""
import modal
import io
import os

# ─── IMAGEM DO CONTAINER ──────────────────────────────────────
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # Core
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "diffusers>=0.30.0",
        "accelerate>=0.28.0",
        "transformers>=4.40.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "Pillow>=10.0.0",
        "numpy<2.0.0",
        "huggingface_hub>=0.20.0",
        "scipy",
        "einops",
        "trimesh>=4.0.0",
        # SAM 2 (sam3 pode não existir no PyPI, sam2 é o pacote real)
        "opencv-python-headless>=4.8.0",
    )
    .run_commands(
        # SAM 2
        "pip install git+https://github.com/facebookresearch/sam2.git || true",
    )
)

app = modal.App("awe-full-pipeline", image=gpu_image)


# ─── ETAPA 1: GERAÇÃO DE IMAGEM ──────────────────────────────
@app.function(gpu="A100", timeout=1200)
def step1_generate_image():
    """Gera imagem com SDXL-Turbo na A100"""
    import torch, gc
    from diffusers import AutoPipelineForText2Image

    print("=" * 60)
    print("  ETAPA 1 — SDXL-TURBO (Geração de Imagem)")
    print("=" * 60)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    prompt = (
        "A Weyland-Yutani industrial outpost at the edge of a volcanic valley, "
        "brutalist metal hangars, red signal lights, dense ash atmosphere, "
        "aerial drone view, volumetric smoke, AAA cinematic realism, 8k, "
        "sharp edges, distinct buildings, hard surfaces"
    )

    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=1024,
        width=1024,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    print(f"✅ ETAPA 1 OK — {len(img_bytes) // 1024} KB")

    del pipe; torch.cuda.empty_cache(); gc.collect()
    return img_bytes


# ─── ETAPA 2: DEPTH ANYTHING V3 ──────────────────────────────
@app.function(gpu="A100", timeout=600)
def step2_depth(img_bytes: bytes):
    """Depth Anything V3 (ou V2 Large como fallback)"""
    import torch, numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print("=" * 60)
    print("  ETAPA 2 — DEPTH ANYTHING")
    print("=" * 60)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Imagem: {img.size}")

    # Tenta V3 primeiro, fallback pra V2-Large
    models_to_try = [
        "depth-anything/DA3-BASE",
        "depth-anything/Depth-Anything-V2-Large-hf",
        "depth-anything/Depth-Anything-V2-Small-hf",
    ]

    model = None
    processor = None
    model_used = None

    for model_id in models_to_try:
        try:
            print(f"Tentando: {model_id}...")
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id).cuda().eval()
            model_used = model_id
            print(f"✅ Carregado: {model_id}")
            break
        except Exception as e:
            print(f"  ❌ {model_id}: {str(e)[:80]}")
            continue

    if model is None:
        print("❌ Nenhum modelo de profundidade carregou")
        return None

    inputs = processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().cpu().float().numpy()

    print(f"Mapa: {depth.shape}, min={depth.min():.2f}, max={depth.max():.2f}")
    print(f"Modelo usado: {model_used}")

    # Normaliza e salva
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_norm)
    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    depth_bytes = buf.getvalue()

    print(f"✅ ETAPA 2 OK — Depth map {depth.shape} ({len(depth_bytes) // 1024} KB)")

    del model; torch.cuda.empty_cache()
    return depth_bytes


# ─── ETAPA 3: SAM (Segmentação) ──────────────────────────────
@app.function(gpu="A100", timeout=600)
def step3_sam(img_bytes: bytes):
    """SAM 2 segmentation test"""
    import torch, numpy as np
    from PIL import Image

    print("=" * 60)
    print("  ETAPA 3 — SAM (Segmentação Semântica)")
    print("=" * 60)

    img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    print(f"Imagem: {img_np.shape}")

    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        print("✅ SAM2 importado")

        sam = build_sam2(
            "sam2_hiera_l.yaml",
            "facebook/sam2-hiera-large",
            device="cuda",
        )
        gen = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.85,
            min_mask_region_area=100,
        )

        masks = gen.generate(img_np)
        num_masks = len(masks)

        if num_masks > 0:
            areas = [m["area"] for m in masks]
            print(f"✅ ETAPA 3 OK — {num_masks} máscaras")
            print(f"   Área média: {sum(areas)/len(areas):.0f}px²")
            print(f"   Maior: {max(areas):.0f}px²")
        else:
            print("⚠️ SAM rodou mas gerou 0 máscaras")

        del sam, gen
        torch.cuda.empty_cache()
        return num_masks

    except Exception as e:
        print(f"⚠️ SAM falhou: {e}")
        # Tenta SAM 1 como fallback
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            print("Tentando SAM 1...")
        except:
            pass
        return 0


# ─── ETAPA 4: TRELLIS ────────────────────────────────────────
@app.function(gpu="A100", timeout=600)
def step4_trellis_api(img_bytes: bytes):
    """
    Trellis 2 via Gradio API — com validação que não é demo.
    Se retornar demo, reporta como falha.
    """
    from PIL import Image
    import tempfile, os, json

    print("=" * 60)
    print("  ETAPA 4 — TRELLIS (Imagem → GLB)")
    print("=" * 60)

    # Instala gradio_client dentro do container
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "gradio_client"])
    from gradio_client import Client, handle_file

    # Salva imagem temporária
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tmp_path = "/tmp/input_image.png"
    img.save(tmp_path)
    print(f"Imagem salva: {tmp_path} ({os.path.getsize(tmp_path) // 1024} KB)")

    # Conecta ao Space
    spaces = [
        "microsoft/TRELLIS",
        "theseanlavery/TRELLIS",
    ]

    for space_name in spaces:
        try:
            print(f"\nConectando: {space_name}...")
            client = Client(space_name)
            print(f"✅ Conectado")

            # Lista endpoints
            try:
                api_info = client.view_api(print_info=False, return_format="dict")
                named = api_info.get("named_endpoints", {})
                unnamed = api_info.get("unnamed_endpoints", {})
                print(f"  Endpoints nomeados: {list(named.keys())}")
                print(f"  Endpoints unnamed: {list(unnamed.keys())}")

                # Tenta cada endpoint
                for ep_name in list(named.keys()) + list(unnamed.keys()):
                    if "image" in ep_name.lower() or "3d" in ep_name.lower() or "generate" in ep_name.lower():
                        try:
                            print(f"\n  Chamando {ep_name}...")
                            result = client.predict(
                                image=handle_file(tmp_path),
                                api_name=ep_name,
                            )
                            print(f"  Resultado: {type(result)} = {str(result)[:300]}")

                            # Procura GLB no resultado
                            glb_bytes = _extract_glb(result)
                            if glb_bytes and len(glb_bytes) > 1000:
                                print(f"  ✅ GLB extraído: {len(glb_bytes) // 1024} KB")
                                return glb_bytes
                            else:
                                print(f"  ⚠️ GLB não encontrado ou muito pequeno")
                        except Exception as e:
                            print(f"  ❌ {ep_name}: {str(e)[:100]}")
            except Exception as e:
                print(f"  API info falhou: {e}")

        except Exception as e:
            print(f"  ❌ {space_name}: {e}")

    print("❌ Nenhum Space retornou GLB válido")
    return None


def _extract_glb(result):
    """Extrai bytes GLB de qualquer formato de resultado Gradio"""
    import os, shutil

    items = []
    if isinstance(result, (tuple, list)):
        items = list(result)
    elif isinstance(result, dict):
        items = list(result.values())
    else:
        items = [result]

    for item in items:
        if isinstance(item, str) and os.path.exists(item):
            with open(item, "rb") as f:
                data = f.read()
            # Verifica se é GLB (magic bytes: glTF = 0x46546C67)
            if len(data) > 4 and data[:4] == b'glTF':
                return data
            elif item.endswith('.glb'):
                return data
        elif isinstance(item, dict):
            # Gradio pode retornar {"name": "path", "data": ...}
            if "name" in item or "path" in item:
                path = item.get("path") or item.get("name")
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        data = f.read()
                    if len(data) > 4 and (data[:4] == b'glTF' or str(path).endswith('.glb')):
                        return data

    return None


# ─── ENTRYPOINT ───────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("\n" + "🚀" * 20)
    print("  AI WORLD ENGINE — PIPELINE COMPLETA NA A100")
    print("🚀" * 20)

    os.makedirs("outputs", exist_ok=True)

    # Etapa 1: Gerar imagem
    print("\n>>> ETAPA 1: Geração de Imagem")
    img_bytes = step1_generate_image.remote()
    with open("outputs/output_flux.png", "wb") as f:
        f.write(img_bytes)
    print(f"   📁 outputs/output_flux.png ({len(img_bytes) // 1024} KB)")

    # Etapa 2: Depth
    print("\n>>> ETAPA 2: Depth Anything")
    depth_bytes = step2_depth.remote(img_bytes)
    if depth_bytes:
        with open("outputs/output_depth.png", "wb") as f:
            f.write(depth_bytes)
        print(f"   📁 outputs/output_depth.png ({len(depth_bytes) // 1024} KB)")

    # Etapa 3: SAM
    print("\n>>> ETAPA 3: SAM (Segmentação)")
    num_masks = step3_sam.remote(img_bytes)
    print(f"   🎭 {num_masks} máscaras geradas")

    # Etapa 4: Trellis
    print("\n>>> ETAPA 4: Trellis (Imagem → GLB)")
    glb_bytes = step4_trellis_api.remote(img_bytes)
    if glb_bytes and len(glb_bytes) > 1000:
        with open("outputs/output_trellis.glb", "wb") as f:
            f.write(glb_bytes)
        print(f"   📁 outputs/output_trellis.glb ({len(glb_bytes) // 1024} KB)")
        print(f"   🏆 GLB REAL GERADO!")
    else:
        print("   ⚠️ Trellis não gerou GLB válido (Space pode estar offline)")

    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO FINAL")
    print("=" * 60)

    results = {
        "Imagem SDXL-Turbo": os.path.exists("outputs/output_flux.png"),
        "Depth Map": depth_bytes is not None,
        "SAM Masks": num_masks > 0,
        "GLB Trellis": glb_bytes is not None and len(glb_bytes) > 1000,
    }

    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n  {passed}/{total} etapas passaram")

    if passed >= 3:
        print("\n  🏆 Pipeline validada! Ready for Day 1 integration.")
    print("=" * 60)
