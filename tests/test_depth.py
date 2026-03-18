"""
ETAPA 4 — Teste do Depth Anything V2 (mapa de profundidade isolado)
Usa a imagem gerada pelo test_flux.py como entrada.
Critério de sucesso: mapa de profundidade gerado, salvo como PNG.
"""
import sys
import os
import torch
import numpy as np

print("=" * 60)
print("ETAPA 4 — DEPTH ANYTHING V2 (Mapa de Profundidade)")
print("=" * 60)

INPUT_IMAGE = "tests/output_flux.png"
OUTPUT_DEPTH = "tests/output_depth.png"

if not os.path.exists(INPUT_IMAGE):
    print(f"❌ Imagem de entrada não encontrada: {INPUT_IMAGE}")
    print("   Execute test_flux.py primeiro.")
    sys.exit(1)

print("\n[1/3] Carregando Depth Anything V2...")
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("❌ transformers não instalado: pip install transformers")
    sys.exit(1)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
model = model.to("cuda").eval()
print(f"  ✅ Modelo carregado: {MODEL_ID}")

print("\n[2/3] Estimando profundidade...")
from PIL import Image
img = Image.open(INPUT_IMAGE).convert("RGB")
inputs = processor(images=img, return_tensors="pt")
inputs = {k: v.to("cuda", dtype=torch.float16) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

depth = outputs.predicted_depth.squeeze().cpu().float().numpy()
print(f"  ✅ Mapa gerado: {depth.shape[1]}x{depth.shape[0]}")
print(f"  📊 Min: {depth.min():.3f} | Max: {depth.max():.3f} | Média: {depth.mean():.3f}")

print("\n[3/3] Salvando visualização...")
depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
depth_img = Image.fromarray(depth_norm)
depth_img.save(OUTPUT_DEPTH)
print(f"  ✅ Depth map salvo: {OUTPUT_DEPTH}")

del model, processor
torch.cuda.empty_cache()
import gc; gc.collect()

print("\n" + "=" * 60)
print("✅ ETAPA 4 CONCLUÍDA — Prosseguir para test_pipeline_minimal.py")
print("=" * 60)
