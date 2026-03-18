#!/bin/bash
# =============================================================================
# AI WORLD ENGINE — SETUP DIA 1 (Do Zero ao GLB)
# Roda UMA VEZ quando a máquina Lightning/RunPod sobe do zero.
# =============================================================================
set -e  # Para na primeira falha

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       AI WORLD ENGINE — SETUP DIA 1                    ║"
echo "║       Do zero ao primeiro GLB real                      ║"
echo "╚══════════════════════════════════════════════════════════╝"

# --------------------------------------------------------------------------
# PASSO 0: Verificações básicas
# --------------------------------------------------------------------------
echo ""
echo "─── PASSO 0: Verificando ambiente ───────────────────────────"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "❌ nvidia-smi falhou. A máquina tem GPU?"
    exit 1
}

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponível'" && \
    echo "✅ PyTorch + CUDA OK" || {
    echo "❌ PyTorch sem CUDA. Instale a versão correta:"
    echo "   pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
}

# --------------------------------------------------------------------------
# PASSO 1: Dependências pip principais
# --------------------------------------------------------------------------
echo ""
echo "─── PASSO 1: Instalando dependências pip ────────────────────"

pip install -q --upgrade pip

pip install -q \
    "diffusers>=0.30.0" \
    "accelerate>=0.28.0" \
    "transformers>=4.40.0" \
    "safetensors>=0.4.0" \
    "huggingface_hub>=0.20.0" \
    "bitsandbytes>=0.43.0" \
    sentencepiece \
    "opencv-python-headless>=4.8.0" \
    "Pillow>=10.0.0" \
    "trimesh>=4.0.0" \
    fast_simplification \
    "numpy<2.0.0" \
    scipy \
    realesrgan \
    triton

echo "✅ Dependências pip instaladas"

# --------------------------------------------------------------------------
# PASSO 2: SAM 3 (Segment Anything Model 3)
# --------------------------------------------------------------------------
echo ""
echo "─── PASSO 2: Instalando SAM 3 ───────────────────────────────"

if python3 -c "import sam3" 2>/dev/null; then
    echo "✅ sam3 já instalado — pulando"
else
    pip install -q git+https://github.com/facebookresearch/sam2.git
    # tenta o nome correto conforme pip list
    pip install -q git+https://github.com/facebookresearch/sam3.git 2>/dev/null || \
        echo "⚠️  sam3 via git falhou — verifique manualmente depois"
fi

python3 -c "import sam3; print('  ✅ sam3:', sam3.__version__ if hasattr(sam3,'__version__') else 'ok')" 2>/dev/null || \
    echo "  ⚠️  sam3 importado mas versão não detectada"

# --------------------------------------------------------------------------
# PASSO 3: Trellis 2
# --------------------------------------------------------------------------
echo ""
echo "─── PASSO 3: Instalando Trellis 2 ──────────────────────────"

if python3 -c "import trellis2" 2>/dev/null; then
    echo "✅ trellis2 já instalado — pulando"
else
    if [ ! -d "trellis2" ]; then
        git clone --depth=1 https://github.com/microsoft/TRELLIS.git trellis2
    fi
    cd trellis2 && pip install -q -e . && cd ..
fi

python3 -c "from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline; print('  ✅ Trellis2ImageTo3DPipeline OK')" || \
    echo "  ❌ Trellis falhou no import — verifique o clone"

# --------------------------------------------------------------------------
# PASSO 4: Download dos pesos (modelos)
# --------------------------------------------------------------------------
echo ""
echo "─── PASSO 4: Baixando pesos dos modelos ─────────────────────"
echo "    (Pode demorar 20-40 min dependendo da conexão)"

python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

models = {
    "FLUX.1-dev":      "black-forest-labs/FLUX.1-dev",
    "DepthAnything":   "depth-anything/Depth-Anything-V2-Small-hf",
    "TRELLIS":         "microsoft/TRELLIS-image-large",
}

# Token HuggingFace (FLUX é gated - precisa de token)
hf_token = os.environ.get("HF_TOKEN", None)
if not hf_token:
    print("⚠️  HF_TOKEN não definido. FLUX.1-dev é gated — pode falhar.")
    print("   Defina: export HF_TOKEN=hf_xxx")

for name, repo_id in models.items():
    print(f"\n  Baixando {name} ({repo_id})...")
    try:
        snapshot_download(
            repo_id=repo_id,
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        print(f"  ✅ {name} — OK")
    except Exception as e:
        print(f"  ❌ {name} falhou: {e}")
PYEOF

# SAM 3 checkpoint
echo ""
echo "  Baixando SAM 3 checkpoint (hiera-large ~2.4GB)..."
python3 -c "
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download('facebook/sam3-hiera-large', filename='sam3_hiera_large.pt')
    print(f'  ✅ SAM 3 checkpoint: {path}')
except Exception as e:
    print(f'  ⚠️  SAM 3 checkpoint falhou: {e}')
    print('     Tente: wget https://dl.fbaipublicfiles.com/segment_anything_3/sam3_hiera_large.pt')
"

# --------------------------------------------------------------------------
# PASSO 5: Clonar o repositório (se necessário)
# --------------------------------------------------------------------------
echo ""
echo "─── PASSO 5: Código do AI World Engine ──────────────────────"

if [ -f "titan_master.py" ]; then
    echo "✅ Repositório já presente — fazendo git pull"
    git pull
else
    echo "Clonando repositório..."
    git clone https://github.com/raiaspnk/aiworldmachineultra.git .
    echo "✅ Repositório clonado"
fi

mkdir -p tests outputs

# --------------------------------------------------------------------------
# PASSO 6: Validação rápida final
# --------------------------------------------------------------------------
echo ""
echo "─── PASSO 6: Validação do setup ────────────────────────────"

python3 tests/test_env.py

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ SETUP CONCLUÍDO!                                    ║"
echo "║                                                          ║"
echo "║  Próximo passo — rodar os testes modulares:             ║"
echo "║    bash tests/run_all.sh                                ║"
echo "║                                                          ║"
echo "║  Se tudo passar → primeiro GLB real do projeto!         ║"
echo "╚══════════════════════════════════════════════════════════╝"
