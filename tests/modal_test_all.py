"""
AI WORLD ENGINE — Teste Modular via Modal (A100 40GB)
Usa SDXL-Turbo (100% aberto, sem login HF)
Roda: modal run tests/modal_test_all.py
"""
import modal
import io
import os

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "diffusers>=0.30.0",
        "accelerate>=0.28.0",
        "transformers>=4.40.0",
        "safetensors>=0.4.0",
        "Pillow>=10.0.0",
        "numpy<2.0.0",
    )
)

app = modal.App("awe-test-v2", image=gpu_image)


@app.function(gpu="A100", timeout=600)
def test_env():
    """ETAPA 0 — Verificação"""
    import sys, torch
    print("=" * 50)
    print("ETAPA 0 — AMBIENTE")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
        print("✅ ETAPA 0 OK")
        return True
    return False


@app.function(gpu="A100", timeout=1200)
def test_image_gen():
    """ETAPA 1 — Gerar imagem com SDXL-Turbo (100% aberto)"""
    import torch, gc
    from diffusers import AutoPipelineForText2Image

    print("=" * 50)
    print("ETAPA 1 — SDXL-TURBO (Modelo Aberto)")
    print("=" * 50)

    print("Carregando SDXL-Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    print("Gerando imagem...")
    image = pipe(
        "A Weyland-Yutani industrial outpost at the edge of a volcanic valley, "
        "brutalist metal hangars, red signal lights, dense ash atmosphere, "
        "aerial drone view, volumetric smoke, AAA cinematic realism",
        num_inference_steps=4,
        guidance_scale=0.0,
        height=1024,
        width=1024,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    print(f"✅ ETAPA 1 OK — imagem gerada ({len(buf.getvalue()) // 1024} KB)")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return buf.getvalue()


@app.function(gpu="A100", timeout=600)
def test_depth(img_bytes: bytes):
    """ETAPA 2 — Depth Anything V2"""
    import torch, numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print("=" * 50)
    print("ETAPA 2 — DEPTH ANYTHING V2")
    print("=" * 50)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Imagem: {img.size}")

    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).cuda()

    inputs = processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        depth = model(**inputs).predicted_depth.squeeze().cpu().numpy()

    print(f"Mapa: {depth.shape}, min={depth.min():.2f}, max={depth.max():.2f}")

    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_norm)

    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    print("✅ ETAPA 2 OK")

    del model
    torch.cuda.empty_cache()
    return buf.getvalue()


@app.local_entrypoint()
def main():
    print("\n🚀 AI WORLD ENGINE — TESTE MODAL A100")
    print("=" * 50)

    # Etapa 0
    print("\n>>> ETAPA 0: Ambiente")
    env_ok = test_env.remote()
    if not env_ok:
        print("❌ Ambiente falhou.")
        return

    # Etapa 1
    print("\n>>> ETAPA 1: Geração de Imagem (SDXL-Turbo)")
    img_bytes = test_image_gen.remote()
    os.makedirs("outputs", exist_ok=True)
    if img_bytes:
        with open("outputs/output_flux.png", "wb") as f:
            f.write(img_bytes)
        print(f"   📁 Imagem salva: outputs/output_flux.png")

    # Etapa 2
    print("\n>>> ETAPA 2: Depth Anything")
    depth_bytes = test_depth.remote(img_bytes)
    if depth_bytes:
        with open("outputs/output_depth.png", "wb") as f:
            f.write(depth_bytes)
        print("   📁 Depth map salvo: outputs/output_depth.png")

    print("\n" + "=" * 50)
    print("🏆 TESTES CONCLUÍDOS!")
    print("   Abra: outputs/output_flux.png")
    print("   Abra: outputs/output_depth.png")
    print("=" * 50)
