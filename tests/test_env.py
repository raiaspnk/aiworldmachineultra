"""
ETAPA 0 — Verificação do Ambiente
Roda PRIMEIRO. Se falhar aqui, não adianta rodar mais nada.
"""
import sys
import subprocess

print("=" * 60)
print("AI WORLD ENGINE — DIAGNÓSTICO DE AMBIENTE")
print("=" * 60)

# Python
print(f"\n[Python] {sys.version}")

# PyTorch + CUDA
try:
    import torch
    print(f"[PyTorch] {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"[CUDA disponível] {cuda_ok}")

    if cuda_ok:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM total] {total_vram:.1f} GB")
    else:
        print("\n❌ PARAR AQUI. CUDA não encontrado.")
        print("Verifique drivers NVIDIA e instalação do PyTorch com suporte CUDA.")
        sys.exit(1)
except ImportError:
    print("❌ PyTorch não instalado.")
    sys.exit(1)

# Bibliotecas críticas
libs = {
    "diffusers": "diffusers",
    "transformers": "transformers",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "safetensors": "safetensors",
}

print("\n[Bibliotecas]")
all_ok = True
for module, pip_name in libs.items():
    try:
        __import__(module)
        print(f"  ✅ {module}")
    except ImportError:
        print(f"  ❌ {module} — instalar: pip install {pip_name}")
        all_ok = False

# SAM 3
print("\n[SAM 3]")
try:
    import sam3
    print("  ✅ sam3 encontrado")
    try:
        from sam3.build_sam import build_sam3
        from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
        print("  ✅ build_sam3, SAM3AutomaticMaskGenerator importados")
    except ImportError as e:
        print(f"  ⚠️  sam3 instalado mas submodule falhou: {e}")
except ImportError:
    print("  ❌ sam3 não encontrado")
    all_ok = False

# Trellis 2
print("\n[Trellis 2]")
try:
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    print("  ✅ trellis2 encontrado")
except ImportError as e:
    print(f"  ❌ trellis2 não encontrado: {e}")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✅ AMBIENTE OK — Prosseguir para test_flux.py")
else:
    print("❌ Corrija os itens acima antes de continuar.")
print("=" * 60)
