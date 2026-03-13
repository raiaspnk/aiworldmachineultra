# ⚡ AI World Engine V11 — Guia de Setup (Lightning.ai Studio)

O Lightning.ai é perfeito para a V11, com suporte nativo a PyTorch avançado e acesso rápido a GPUs parrudas (A100, L40S, RTX 4090, RTX 6000 Ada). Siga os passos abaixo no terminal do seu **Studio** recém-criado.

## 0. Criar o Studio e Escolher a GPU

1. Acesse o **Lightning.ai**.
2. Clique em **Start a new Studio**.
3. Escolha o template em branco (Blank Studio) com **PyTorch habilitado**.
4. No menu superior direito (Hardware), ligue uma máquina de peso pesado (Recomendado: **L40S, A100 ou RTX 4090/6000 Ada** com pelo menos 24GB+ de VRAM).
   * *O projeto V11 exige bastante VRAM para aguentar o FLUX e o TRELLIS 2, portanto foque em 24GB ou mais.*

---

## 1. Subir o Projeto V11

Você precisa colocar a pasta do AI World Engine dentro da raiz do Studio (normalmente em `/workspace`).
Abra o terminal do Studio e clone/descompacte seu repositório:

```bash
cd /workspace
# Clonando o repositório V11:
git clone https://github.com/raiaspnk/aiworldmachineultra.git
cd aiworldmachineultra

# Ou, arraste a pasta local do seu Windows
# para dentro da interface web do Lightning Studio.
```

---

## 2. Preparar o Ambiente Virtual Python

O Lightning Studio já vem com Python e PyTorch, mas é melhor criar um ambiente isolado para evitar conflitos:

```bash
# Atualize o apt e instale essenciais de build (importante para compilar o C++)
sudo apt-get update
sudo apt-get install -y build-essential curl ninja-build ffmpeg libsm6 libxext6 libgl1 

# Instalar o xformers e PyTorch 2.4.0+ (caso o Studio esteja desatualizado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Instalação das Dependências I.A. (Requirements)

Instale os pacotes core de IA e processamento de imagem/geometria:

```bash
# O projeto possui um arquivo de requirements dedicado para GPU:
pip install -r requirements_gpu.txt

# (Apenas caso algum pacote não esteja no arquivo, instale manualmente:)
# pip install segment_anything_hq depth_anything_v2 supervision trimesh fast-simplification
```

> [!TIP]
> **Autenticação no HuggingFace:** O FLUX.2-dev e o TRELLIS 2 exigem token de acesso.
> Execute o comando: `huggingface-cli login`
> E cole o seu token público do HuggingFace.

---

## 4. Instalar o TRELLIS 2 (AssetForge)

O AssetForge depende intensamente do algoritmo TRELLIS da Microsoft para gerar geometrias a partir do POVs:

```bash
# Acesse a engine
cd /workspace/aiworldmachineultra

# Baixando e instalando o pacote Trellis oficial:
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
pip install -e .
cd ..
```

*Nota: Durante o boot, a V11 (`titan_master.py`) vai baixar automaticamente os modelos pesados para o seu cache (na pasta `~/.cache/huggingface/`).*

---

## 5. Compilar os Kernels C++ Custom (Opcional, mas recomendado)

O projeto V10/V11 possui scripts C++ com instruções PTX customizadas (`socket_engine.cu`) para simulação de Física (catenária) agressivas sem estressar a GPU.

**Como compilar o PyBind11 no Lightning:**
```bash
cd /workspace/aiworldmachineultra/kernels

# Instalando dependência pra linkar o Python com C/CUDA
pip install pybind11

# Você pode usar um simples compilar do NVCC se já tiver criado um setup.py (se você tiver ele)
# nvidia-smi para confirmar a versão do CUDA (deve ser > 11.8)
nvcc -O3 -shared -Xcompiler -fPIC -I $(python3 -m pybind11 --includes) socket_engine.cu -o socket_engine_cuda.so

cd ..
```

---

## 6. Ligar o Orquestrador! (Rodar a V11)

Está tudo pronto! Agora vamos testar a Fase Macro gerando o Master Plan e esculpindo as cidades:

Você pode inicializar via terminal ou criar um script de boot:

```bash
python titan_master.py --prompt "Cidade cyberpunk japonesa neon e destruida, clima noturno, chuva, cyberware, ruas estreitas" --seed 42 --use-v11-macro
```

> [!IMPORTANT]
> O código fará um **cold start** de alguns minutos enquanto baixa 15GB+ de modelos (FLUX, SAM3, TRELLIS2, Real-ESRGAN). 
> Nas próximas rodadas o `_warmup_models()` já entrará em ação com Cache Hit e pulará essa parte.

---

## Dicas Finais do Lightning.ai

1. **Persistência de Arquivos:** Tudo dentro de `/workspace/` no Lightning fica salvo se a máquina for desligada. Não crie arquivos pesados fora disso!
2. **Monitoramento:** Enquanto o `titan_master.py` rodar, abra outra aba e digite `watch -n 1 nvidia-smi` para monitorar alegremente a RAM e GPU fritando em 400w puxando as texturas.
3. **Download Rápido:** Após exportar o glorioso Mundo em `.glb` no `output/world_output.glb`, use a interface Web do Lightning para abaixá-lo usando o botão direito no painel lateral de arquivos -> "Download". (Se for muito grande, o `exportador multipart OOM` do Batch 8 já ajudará!)
