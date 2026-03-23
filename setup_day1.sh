#!/bin/bash
set -e

echo "=========================================================="
echo "    AI WORLD ENGINE - DAY 1 LIGHTNING SETUP SCRIPT        "
echo "=========================================================="

# FUNDAMENTAL PARA A LIGHTNING: Salvar os pesos gigas na pasta que NÃO apaga no restart
export HF_HOME="/workspace/.cache/huggingface"
echo "📦 HF_HOME = $HF_HOME (Para evitar re-download de 40GB do FLUX.2 toda vez)"

echo "[1/6] Atualizando pacotes OS (libgl1 para OpenCV GUI/Real-ESRGAN)..."
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 git build-essential ninja-build || echo "⚠️ Aviso: apt-get falhou (talvez nao seja root), continuando..."

echo "[2/6] Instalando dependencias base (Ninja, Torch, Diffusers/Transformers @main)..."
pip install -r requirements_gpu.txt

echo "[3/6] Instalando Triton (APOS o Torch)..."
# O Triton precisa ser instalado apos o Torch para que ele herde os 
# binarios corretos do driver CUDA da L40S.
# Se instalar antes do Torch ou solto, o SAM 3 fara fallback para CPU.
pip install triton

echo "[4/6] Instalando SAM 3..."
pip install git+https://github.com/facebookresearch/sam3.git

echo "[5/6] Instalando Depth Anything 3..."
pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git

echo "[6/6] Instalando TRELLIS 2 (Ninja vai compilar os kernels rapido)..."
if [ ! -d "TRELLIS.2" ]; then
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git
fi
cd TRELLIS.2
pip install -e .
cd ..

echo "=========================================================="
echo "    SETUP CONCLUIDO COM SUCESSO!                          "
echo "    Certifique-se que HF_TOKEN esta exportado.            "
echo "    Execute: bash tests/run_all.sh para testar os modulos."
echo "=========================================================="
