"""
ETAPA 2 — Teste do SAM 3 (segmentação isolada)
Usa a imagem gerada pelo test_flux.py como entrada.
Critério de sucesso: pelo menos 1 máscara gerada, sem crash.
"""
import sys
import os
import numpy as np

# Garante que o sam3 local seja encontrado
sys.path.insert(0, os.path.abspath("sam3"))
sys.path.insert(0, os.path.abspath("."))

print("=" * 60)
print("ETAPA 2 — SAM 3 (Segmentação Semântica)")
print("=" * 60)

INPUT_IMAGE = "tests/output_flux.png"
if not os.path.exists(INPUT_IMAGE):
    print(f"❌ Imagem de entrada não encontrada: {INPUT_IMAGE}")
    print("   Execute test_flux.py primeiro.")
    sys.exit(1)

from PIL import Image
img_pil = Image.open(INPUT_IMAGE).convert("RGB")
img_np = np.array(img_pil)
print(f"\n[1/3] Imagem carregada: {img_np.shape[1]}x{img_np.shape[0]}px")

print("\n[2/3] Carregando SAM 3...")
try:
    from sam3.build_sam import build_sam3
    from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
except ImportError as e:
    print(f"❌ Falha ao importar sam3: {e}")
    print("   Verifique: pip list | grep sam3")
    sys.exit(1)

sam = build_sam3(
    config_file="sam3_hiera_l.yaml",
    ckpt_path="facebook/sam3-hiera-large",
    device="cuda",
)
print("  ✅ Modelo SAM 3 carregado")

generator = SAM3AutomaticMaskGenerator(
    model=sam,
    points_per_side=16,       # Reduzido para teste rápido
    pred_iou_thresh=0.80,
    stability_score_thresh=0.85,
    min_mask_region_area=100,
)

print("\n[3/3] Gerando máscaras...")
masks = generator.generate(img_np)

print(f"  ✅ {len(masks)} máscaras geradas")
if len(masks) > 0:
    areas = [m["area"] for m in masks]
    print(f"  📊 Área média: {sum(areas)/len(areas):.0f}px²")
    print(f"  📊 Maior máscara: {max(areas):.0f}px²")

import torch, gc
del sam, generator
torch.cuda.empty_cache()
gc.collect()

print("\n" + "=" * 60)
if len(masks) > 0:
    print("✅ ETAPA 2 CONCLUÍDA — Prosseguir para test_trellis.py")
else:
    print("⚠️  SAM 3 carregou mas gerou 0 máscaras. Verifique a imagem.")
print("=" * 60)
