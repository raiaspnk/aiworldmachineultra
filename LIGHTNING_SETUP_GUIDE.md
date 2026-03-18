# ⚡ AI World Engine — Guia de Setup Dia 1 (Lightning.ai)

> **Versão:** V12 | **GPU alvo:** L40S (48 GB VRAM) | **Python:** 3.10+

---

## 0. Criar o Studio

1. Acesse **lightning.ai** → **New Studio**
2. Template: **Blank** com PyTorch habilitado
3. Hardware: **L40S** (48 GB VRAM) — mínimo A100 ou RTX 4090

---

## 1. Abrir o Terminal e Clonar o Projeto

```bash
cd /workspace

git clone https://github.com/raiaspnk/aiworldmachineultra.git
cd aiworldmachineultra
```

---

## 2. Instalar Dependências do Sistema

```bash
sudo apt-get update -q
sudo apt-get install -y build-essential ninja-build ffmpeg libsm6 libxext6 libgl1
```

---

## 3. PyTorch + CUDA (se o Studio estiver desatualizado)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verificação — deve imprimir True
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. Dependências Python

```bash
pip install -r requirements_gpu.txt
```

---

## 5. Autenticar no HuggingFace (obrigatório para FLUX)

```bash
# Opção A — interativo
huggingface-cli login

# Opção B — variável de ambiente (mais rápido)
export HF_TOKEN=hf_SEU_TOKEN_AQUI
```

> Pegue seu token em: https://huggingface.co/settings/tokens

---

## 6. Instalar SAM 3

```bash
pip install git+https://github.com/facebookresearch/sam3.git

# Verifica
python -c "import sam3; print('SAM 3 OK')"
```

---

## 7. Instalar Trellis 2

```bash
git clone --depth=1 https://github.com/microsoft/TRELLIS.git trellis2
cd trellis2 && pip install -e . && cd ..

# Verifica
python -c "from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline; print('Trellis 2 OK')"
```

---

## 8. Rodar o Diagnóstico Modular (A Ordem Importa!)

> ⚠️ **Não rode o `titan_master.py` antes de todos os testes passarem.**

```bash
# Testa cada módulo isolado — para na primeira falha
bash tests/run_all.sh
```

O script vai rodar em sequência:

| Script | O que valida | Arquivo gerado |
|---|---|---|
| `test_env.py` | CUDA, PyTorch, todas as libs | — |
| `test_flux.py` | FLUX.1-dev gera imagem | `tests/output_flux.png` |
| `test_sam3.py` | SAM 3 fatia a imagem | — |
| `test_trellis.py` | **Trellis gera o GLB** | `tests/output_trellis.glb` ⭐ |
| `test_depth.py` | Depth Anything funciona | `tests/output_depth.png` |

---

## 9. Resultado Esperado

```
✅ ETAPA 0: Ambiente OK
✅ ETAPA 1: FLUX OK → tests/output_flux.png
✅ ETAPA 2: SAM 3 OK → X máscaras geradas
✅ ETAPA 3: TRELLIS OK → tests/output_trellis.glb  ← PRIMEIRO GLB REAL!
✅ ETAPA 4: DEPTH OK → tests/output_depth.png
```

Abra o `output_trellis.glb` no [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com/) para visualizar.

---

## 10. Só Então: Rodar a Pipeline Completa

```bash
python titan_master.py \
  --prompt "Um posto avançado da corporação Weyland-Yutani em um vale vulcânico, hangares industriais, luzes vermelhas, fumaça volumétrica, realismo cinematográfico AAA" \
  --seed 777
```

---

## Dicas Lightning.ai

| Dica | Detalhe |
|---|---|
| **Persistência** | Salve tudo dentro de `/workspace/` |
| **Monitor de GPU** | `watch -n 1 nvidia-smi` em outra aba |
| **Download do GLB** | Painel lateral de arquivos → botão direito → Download |
| **VRAM pós-teste** | Cada `test_*.py` libera a VRAM antes de sair |
