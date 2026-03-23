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

## 2. Instalação Completa (A Usina)

Nós consolidamos todas as instalações (sistema, ninja, pacotes OS, dependências PyTorch, Diffusers/Transformers @main, SAM 3, Trellis) em um único script blindado contra conflitos da GPU L40S.

```bash
export HF_TOKEN=hf_SEU_TOKEN_AQUI
bash setup_day1.sh
```

> **Por que usar o script e não na mão?** O `setup_day1.sh` garante a ordem cirúrgica:
> 1. Instala o `ninja-build` para compilar os kernels do Trellis 2x mais rápido.
> 2. Puxa os diffusers e transformers novos (FLUX 2).
> 3. Instala o `triton` **depois** do PyTorch, garantindo que o SAM 3 conecte direto com os binários do CUDA e não faça fallback de memória da L40S para CPU.

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
