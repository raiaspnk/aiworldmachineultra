#!/bin/bash
# ==============================================================================
# AI WORLD ENGINE V5 + MONSTERCORE V5 - AUTO INSTALLER (Multi-Cloud Universal)
# Auto-detecta: Google Cloud, Lightning AI, RunPod, Lambda, Vast.ai
# Testado em: Debian Trixie (Python 3.13), Ubuntu 22/24, RunPod Ubuntu
# ==============================================================================

set -e # Sai do script se qualquer comando falhar

# ── DETECÇÃO AUTOMÁTICA DO AMBIENTE ──────────────────────────────────
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
PIP_FLAGS=""

# ── PYTHON 3.11 (compatibilidade total com pacotes ML) ───────────────
# Python 3.12+ quebra open3d, basicsr, xformers. Usamos 3.11 no venv.
VENV_DIR="$HOME/venv311"
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Configurando Python 3.11 (compatibilidade ML total)..."
    # Instalar python3.11 se não existir
    if ! python3.11 --version &>/dev/null; then
        sudo apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils 2>/dev/null || \
        sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
    fi
    python3.11 -m venv "$VENV_DIR"
    echo "✅ venv311 criado em $VENV_DIR"
fi

# Ativar o venv311
source "$VENV_DIR/bin/activate"
echo "✅ Python ativo: $(python --version)"
PIP_FLAGS=""  # Dentro do venv, não precisa --break-system-packages

# Detectar Cloud Provider
if [ -f /etc/google_cloud ]; then
    CLOUD="Google Cloud"
elif [ -d /teamspace ]; then
    CLOUD="Lightning AI"
elif [ -d /workspace ] && grep -q "runpod" /etc/hostname 2>/dev/null; then
    CLOUD="RunPod"
else
    CLOUD="Genérico"
fi

echo "==================================================================="
echo "🚀 AI World Engine V5 Auto-Setup"
echo "   📍 Cloud: $CLOUD"
echo "   📂 Workspace: $WORKSPACE_DIR"
echo "   🐍 Python: $(python3 --version 2>&1)"
echo "==================================================================="

# ── 1. PACOTES DO SISTEMA ────────────────────────────────────────────
echo "[1/6] 🛠️ Instalando pacotes de sistema e Ninja Build..."
sudo apt-get update -y
# libgl1 (substitui libgl1-mesa-glx no Trixie), libglib2.0-0 auto-resolve pra 0t64
sudo apt-get install -y libgl1 libglib2.0-0 libgomp1 cmake build-essential ninja-build unzip git-lfs wget || true
git lfs install

# ── 2. CLONAR MOTORES DE IA ──────────────────────────────────────────
echo "[2/6] 📥 Clonando os Motores de IA..."
cd "$WORKSPACE_DIR"

if [ ! -d "Real-ESRGAN" ]; then git clone https://github.com/xinntao/Real-ESRGAN.git; fi
if [ ! -d "Hunyuan3D-2-main" ]; then
    git clone https://github.com/Tencent/Hunyuan3D-2.git
    mv Hunyuan3D-2 Hunyuan3D-2-main
fi
if [ ! -d "StableNormal" ]; then git clone https://github.com/Stable-X/StableNormal.git; fi
if [ ! -d "HunyuanWorld-Mirror-main" ]; then
    git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git
    mv HunyuanWorld-Mirror HunyuanWorld-Mirror-main
fi
if [ ! -d "sd-scripts" ]; then git clone https://github.com/kohya-ss/sd-scripts.git; fi

# 2.5 Injetar scripts customizados nos motores clonados
if [ -d "custom_hw_scripts" ]; then
    echo "[2.5/6] 🧩 Injetando scripts customizados..."
    cp custom_hw_scripts/infer.py HunyuanWorld-Mirror-main/infer.py 2>/dev/null || true
    cp -r custom_hw_scripts/src/* HunyuanWorld-Mirror-main/src/ 2>/dev/null || true
    cp custom_hw_scripts/scene_mesh_generator.py Hunyuan3D-2-main/scene_mesh_generator.py 2>/dev/null || true
fi

# ── 3. PYTHON + PYTORCH + CUDA ───────────────────────────────────────
echo "[3/6] 🔥 Instalando PyTorch c/ CUDA 12.4..."
pip install $PIP_FLAGS torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install $PIP_FLAGS xformers --index-url https://download.pytorch.org/whl/cu124

echo "[3.5/6] 📦 Instalando ecossistema Python..."
pip install $PIP_FLAGS -r requirements.txt 2>/dev/null || echo "requirements.txt pulado"
pip install $PIP_FLAGS -r requirements-extra.txt 2>/dev/null || echo "requirements-extra.txt pulado"
pip install $PIP_FLAGS ninja pyvista trimesh open3d pymeshlab basicsr timm transformers scipy xatlas huggingface_hub fastapi uvicorn

# 3.6 Compilar Hunyuan3D-2 (hy3dgen)
echo "[3.6/6] 🏗️ Compilando Hunyuan3D-2 (hy3dgen)..."
cd "$WORKSPACE_DIR/Hunyuan3D-2-main"
pip install $PIP_FLAGS --no-build-isolation -e .

# 3.7 Custom Rasterizer
echo "[3.7/6] 🎨 Compilando Custom Rasterizer..."
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install $PIP_FLAGS 2>/dev/null || echo "⚠️ custom_rasterizer falhou (texturas podem ficar brancas)"

# Voltar pro workspace (path dinâmico, não hardcoded)
cd "$WORKSPACE_DIR"

# ── 4. COMPILAR MONSTERCORE V5 ───────────────────────────────────────
echo "[4/6] ⚙️ Compilando MonsterCore V5 (C++/CUDA)..."
if [ -f "setup.py" ] && [ -f "monster_core.cpp" ] && [ -f "monster_core_kernels.cu" ]; then
    chmod +x bridge.py setup.py
    python3 setup.py build_ext --inplace
    echo "✅ MonsterCore V5 Compilado!"
else
    echo "⚠️ AVISO: Arquivos do MonsterCore não encontrados. Envie setup.py, .cpp e .cu!"
fi

# ── 5. BAIXAR PESOS DAS IAs ─────────────────────────────────────────
echo "[5/6] 🧠 Baixando pesos dos modelos (pode levar alguns minutos)..."
mkdir -p "$WORKSPACE_DIR/weights" "$WORKSPACE_DIR/output" "$WORKSPACE_DIR/models" "$WORKSPACE_DIR/parts" "$WORKSPACE_DIR/sessions"

if [ ! -f "$WORKSPACE_DIR/weights/RealESRGAN_x4plus.pth" ]; then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P "$WORKSPACE_DIR/weights/"
fi

python3 -c "
import os
from huggingface_hub import snapshot_download
print('Baixando Hunyuan3D-2 do HuggingFace...')
wk_dir = os.getcwd()
snapshot_download('tencent/Hunyuan3D-2', local_dir=f'{wk_dir}/weights/Hunyuan3D-2', resume_download=True)
"

# ── 6. TESTE DE IGNIÇÃO ─────────────────────────────────────────────
echo "[6/6] 🏁 Testando ignição do MonsterCore..."
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NENHUMA GPU')" 2>/dev/null || echo "ERRO")
GPU_VRAM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.0f}GB') if torch.cuda.is_available() else print('?')" 2>/dev/null || echo "?")

if python3 -c "import torch; import monster_core; monster_core.init_pool(8192); print('\n✅ MONSTERCORE V5 ONLINE!');" 2>/dev/null; then
    echo "==================================================================="
    echo "🏎️  AI WORLD ENGINE V5 ESTÁ PRONTA!"
    echo "   📍 Cloud: $CLOUD"
    echo "   🎮 GPU: $GPU_NAME ($GPU_VRAM VRAM)"
    echo ""
    echo "   Comando V5 básico:"
    echo "   python3 bridge.py --prompt 'cyberpunk city' --world-mode --use-depth --use-sam"
    echo "==================================================================="
else
    echo "==================================================================="
    echo "⚠️ SETUP CONCLUIDO, MAS O MONSTERCORE NÃO IMPORTOU!"
    echo "   GPU detectada: $GPU_NAME ($GPU_VRAM)"
    echo "   Verifique os logs de compilação acima."
    echo "==================================================================="
fi
