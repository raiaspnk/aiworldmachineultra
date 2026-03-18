"""
ETAPA 3 — Teste do Trellis 2 (imagem → GLB isolado)
USA a imagem gerada pelo test_flux.py como entrada.
Critério de sucesso: salva tests/output_trellis.glb
Este é o primeiro resultado 3D real do projeto.
"""
import sys
import os
import torch
import gc

print("=" * 60)
print("ETAPA 3 — TRELLIS 2 (Image → GLB)")
print("=" * 60)

INPUT_IMAGE = "tests/output_flux.png"
OUTPUT_GLB = "tests/output_trellis.glb"

if not os.path.exists(INPUT_IMAGE):
    print(f"❌ Imagem de entrada não encontrada: {INPUT_IMAGE}")
    print("   Execute test_flux.py primeiro.")
    sys.exit(1)

print("\n[1/3] Carregando Trellis 2...")
try:
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
except ImportError as e:
    print(f"❌ Falha ao importar trellis2: {e}")
    print("   Verifique: pip list | grep trellis")
    sys.exit(1)

pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipe.cuda()
print("  ✅ Pipeline Trellis 2 carregada")

print("\n[2/3] Gerando geometria 3D...")
from PIL import Image
img = Image.open(INPUT_IMAGE).convert("RGB")

with torch.no_grad():
    outputs = pipe.run(
        img,
        seed=777,
        sparse_structure_sampler_params={"steps": 12},
        slat_sampler_params={"steps": 12},
    )

print("  ✅ Geometria gerada")

print("\n[3/3] Exportando GLB...")
glb_bytes = outputs["mesh"][0].export_glb()
with open(OUTPUT_GLB, "wb") as f:
    f.write(glb_bytes)

size_kb = os.path.getsize(OUTPUT_GLB) / 1024
print(f"  ✅ GLB salvo: {OUTPUT_GLB} ({size_kb:.1f} KB)")

del pipe, outputs
torch.cuda.empty_cache()
gc.collect()

print("\n" + "=" * 60)
print("🏆 ETAPA 3 CONCLUÍDA — PRIMEIRO GLB REAL DO PROJETO!")
print(f"   Abra o arquivo: {os.path.abspath(OUTPUT_GLB)}")
print("   (Blender, Windows 3D Viewer, ou online em https://gltf-viewer.donmccurdy.com/)")
print("=" * 60)
