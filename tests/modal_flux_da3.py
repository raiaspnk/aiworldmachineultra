"""
AI WORLD ENGINE — FLUX.1-dev + Depth Anything 3 (Modal A100)
Os modelos REAIS do projeto, não substitutos.
Roda: $env:HF_TOKEN="hf_xxx"; modal run tests/modal_flux_da3.py
"""
import modal
import io
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─── IMAGEM COM FLUX + DA3 ────────────────────────────────────
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .env({"HF_TOKEN": HF_TOKEN})
    .pip_install(
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
        "xformers",
        "opencv-python-headless>=4.8.0",
    )
    .run_commands(
        # Depth Anything 3
        "pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git || "
        "pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git || true",
    )
)

app = modal.App("awe-flux-da3", image=gpu_image)


# ─── FLUX.1-DEV ──────────────────────────────────────────────
@app.function(gpu="A100", timeout=1800)
def test_flux_dev():
    """FLUX.1-dev — o modelo real do projeto"""
    import torch, gc, os
    from huggingface_hub import login

    print("=" * 60)
    print("  FLUX.1-DEV — O Modelo Real")
    print("=" * 60)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

    # Autentica
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
        print("✅ HuggingFace autenticado")
    else:
        print("⚠️ HF_TOKEN não definido — FLUX.1-dev pode falhar")

    from diffusers import FluxPipeline

    # Tenta FLUX.1-dev, fallback pra schnell
    models = [
        ("black-forest-labs/FLUX.1-dev", 28, 3.5),
        ("black-forest-labs/FLUX.1-schnell", 4, 0.0),
    ]

    for model_id, steps, guidance in models:
        try:
            print(f"\nCarregando {model_id}...")
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            ).to("cuda")

            print(f"✅ {model_id} carregado!")
            print(f"Gerando imagem 1024x1024 ({steps} steps)...")

            prompt = (
                "A Weyland-Yutani industrial outpost at the edge of a volcanic valley, "
                "brutalist metal hangars with weathered steel panels, red signal lights pulsing, "
                "dense ash atmosphere, heavy cargo ships parked on landing pads, "
                "aerial drone view, volumetric smoke, AAA cinematic realism, 8k resolution, "
                "sharp edges, distinct buildings, hard industrial surfaces"
            )

            image = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=1024,
                width=1024,
            ).images[0]

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            print(f"✅ Imagem gerada: {len(img_bytes) // 1024} KB")
            print(f"   Modelo usado: {model_id}")

            del pipe; torch.cuda.empty_cache(); gc.collect()
            return img_bytes, model_id

        except Exception as e:
            print(f"❌ {model_id}: {e}")
            try:
                del pipe
            except:
                pass
            torch.cuda.empty_cache(); gc.collect()
            continue

    print("❌ Nenhum modelo FLUX carregou")
    return None, None


# ─── DEPTH ANYTHING 3 ────────────────────────────────────────
@app.function(gpu="A100", timeout=600)
def test_da3(img_bytes: bytes):
    """Depth Anything 3 — o modelo real"""
    import torch, numpy as np
    from PIL import Image

    print("=" * 60)
    print("  DEPTH ANYTHING 3")
    print("=" * 60)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Imagem: {img.size}")

    # Tenta DA3 primeiro
    da3_loaded = False
    try:
        from depth_anything_3 import DepthAnything3
        print("Carregando DA3-BASE...")
        model = DepthAnything3.from_pretrained("depth-anything/da3-base")
        model = model.cuda().eval()
        da3_loaded = True
        model_used = "DA3-BASE"
        print("✅ Depth Anything 3 carregado!")

        # DA3 usa interface própria
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            depth = model(input_tensor)
        if isinstance(depth, dict):
            depth = depth.get("depth", list(depth.values())[0])
        depth = depth.squeeze().cpu().float().numpy()

    except Exception as e:
        print(f"⚠️ DA3 falhou: {e}")

    # Fallback: Depth Anything V2 Large via transformers
    if not da3_loaded:
        print("\nFallback: Depth Anything V2-Large...")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        models_to_try = [
            "depth-anything/Depth-Anything-V2-Large-hf",
            "depth-anything/Depth-Anything-V2-Small-hf",
        ]

        for model_id in models_to_try:
            try:
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForDepthEstimation.from_pretrained(model_id).cuda().eval()
                model_used = model_id
                print(f"✅ {model_id} carregado")

                inputs = processor(images=img, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    depth = model(**inputs).predicted_depth.squeeze().cpu().float().numpy()
                break
            except Exception as e2:
                print(f"  ❌ {model_id}: {str(e2)[:80]}")

    print(f"Mapa: {depth.shape}, min={depth.min():.2f}, max={depth.max():.2f}")
    print(f"Modelo: {model_used}")

    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_norm)

    # Resize depth to match original image
    depth_img = depth_img.resize(img.size, Image.BILINEAR)

    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    depth_bytes = buf.getvalue()

    print(f"✅ Depth map: {len(depth_bytes) // 1024} KB")

    del model; torch.cuda.empty_cache()
    return depth_bytes, model_used


# ─── ENTRYPOINT ───────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("\n" + "🔥" * 20)
    print("  AI WORLD ENGINE — FLUX.1-DEV + DA3 NA A100")
    print("🔥" * 20)

    os.makedirs("outputs", exist_ok=True)

    # FLUX.1-dev
    print("\n>>> ETAPA 1: FLUX.1-dev (o modelo real!)")
    result = test_flux_dev.remote()
    img_bytes, flux_model = result

    if img_bytes:
        with open("outputs/output_flux_dev.png", "wb") as f:
            f.write(img_bytes)
        print(f"   📁 outputs/output_flux_dev.png ({len(img_bytes) // 1024} KB)")
        print(f"   🎨 Modelo: {flux_model}")
    else:
        print("   ❌ FLUX falhou — usando imagem anterior")
        with open("outputs/output_flux.png", "rb") as f:
            img_bytes = f.read()

    # Depth Anything 3
    print("\n>>> ETAPA 2: Depth Anything 3")
    depth_bytes, depth_model = test_da3.remote(img_bytes)

    if depth_bytes:
        with open("outputs/output_depth_v3.png", "wb") as f:
            f.write(depth_bytes)
        print(f"   📁 outputs/output_depth_v3.png ({len(depth_bytes) // 1024} KB)")
        print(f"   📐 Modelo: {depth_model}")

    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"  FLUX:  {'✅ ' + flux_model if flux_model else '❌'}")
    print(f"  Depth: {'✅ ' + depth_model if depth_model else '❌'}")
    print(f"\n  Arquivos em outputs/")
    print("=" * 60)
