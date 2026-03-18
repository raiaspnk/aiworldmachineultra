"""
ETAPA 1 — Teste do FLUX.1-dev (geração de imagem isolada)
Só o modelo de difusão. Nada mais.
Critério de sucesso: salva tests/output_flux.png
"""
import torch
import os

print("=" * 60)
print("ETAPA 1 — FLUX.1-dev")
print("=" * 60)

os.makedirs("tests", exist_ok=True)

from diffusers import FluxPipeline

print("\n[1/3] Carregando FLUX.1-dev...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
print("  ✅ Modelo carregado")

print("\n[2/3] Gerando imagem...")
prompt = (
    "A Weyland-Yutani industrial outpost at the edge of a volcanic valley, "
    "brutalist metal hangars, red signal lights, dense ash atmosphere, "
    "aerial drone view, volumetric smoke, AAA cinematic realism, 8k"
)

with torch.no_grad():
    result = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024,
        width=1024,
    )

image = result.images[0]
output_path = "tests/output_flux.png"
image.save(output_path)
print(f"  ✅ Imagem salva: {output_path}")

# Verifica tamanho do arquivo
size_kb = os.path.getsize(output_path) / 1024
print(f"  📁 Tamanho: {size_kb:.1f} KB")

print("\n[3/3] Limpando VRAM...")
del pipe
torch.cuda.empty_cache()
import gc; gc.collect()
vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
print(f"  ✅ VRAM liberada — {vram_free:.1f} GB disponíveis")

print("\n" + "=" * 60)
print("✅ ETAPA 1 CONCLUÍDA — Prosseguir para test_sam3.py")
print("=" * 60)
