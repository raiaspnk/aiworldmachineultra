# 🚀 GUIA DE SOBREVIVÊNCIA: V6 TITAN NA L40S (LINUX)
**A Colinha de Elite para Configuração do Zero em Instâncias Alugadas (RunPod / Lambda Labs / Vast.ai)**

Este guia garante que você tenha um ambiente 100% livre de conflitos entre os drivers da NVIDIA, CUDA 12.4, PyTorch e o nosso Kernel em C++ (Socket Engine).

---

## 1. O Ponto de Partida (Baseline)
Sempre alugue uma máquina **Ubuntu 22.04 LTS**.
Assim que o SSH conectar, verifique o que a provedora te entregou:

```bash
# Verifica se a placa de vídeo foi reconhecida pelo kernel do Linux
nvidia-smi

# Verifica se os compiladores C++ básicos do Linux estão instalados
gcc --version
g++ --version
```
> ⚠️ **Atenção:** Se o `gcc` não existir, instale com: `sudo apt update && sudo apt install build-essential cmake -y`

---

## 2. A Fundação Conda (O Isolamento Perfeito)
Nunca instale bibliotecas Python no sistema base da máquina alugada. Você vai quebrar algo. Use o Miniconda.

```bash
# 1. Baixe o Miniconda instalador silencioso
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 2. Execute a instalação (-b = batch mode, -p = path)
bash miniconda.sh -b -p $HOME/miniconda

# 3. Injete o Conda no seu terminal atual
source $HOME/miniconda/bin/activate
conda init bash
source ~/.bashrc

# 4. Crie o Universo da V6 Titan (Python 3.10 é o mais estável para os pesos atuais)
conda create -n titan_v6 python=3.10 -y
conda activate titan_v6
```

---

## 3. O Casamento Crítico: PyTorch 2.4 + CUDA 12.4
Este é o ponto onde 90% dos desenvolvedores falham. Se o CUDA do PyTorch não bater com o CUDA do NVCC (compilador C++ da Nvidia), o nosso `socket_engine.cu` vai compilar, mas vai dar *Segmentation Fault* na hora de rodar.

```bash
# Instala o PyTorch 2.4 amarrado EXATAMENTE com o CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

> **Verificação Mestra:**
> Digite `python -c "import torch; print(torch.version.cuda)"`
> *Deve imprimir `12.4`.*

---

## 4. O Kit de Batalha C++ (CUDA Toolkit)
Mesmo instalando o PyTorch com suporte a CUDA 12.4, você precisa do compilador **Nativo** do CUDA (`nvcc`) na mesma versão para montar o `setup.py` do nosso *Socket Engine*.

A forma mais segura de ter o NVCC sem foder o driver da placa de vídeo (que já vem instalado pelo provedor) é instalá-lo DENTRO do Conda:

```bash
# Instala o compilador CUDA 12.4 isolado apenas para o seu ambiente titan_v6
conda install -c nvidia cuda-compiler=12.4.0 cuda-toolkit=12.4.0 -y
```

> **Verificação Mestra:**
> Digite `nvcc --version`
> *Deve mostrar a versão 12.4 e pertencer ao diretório do seu miniconda.*

---

## 5. Clonagem e Compilação do Músculo (PyBind11)
Agora que o ambiente está blindado, vamos compilar o nosso C++ para gerar o `.so` do Linux otimizado para a arquitetura Ada Lovelace.

```bash
# Vá para a pasta do projeto
cd /workspace/AI_World_Engine

# Instale os requisitos estruturais (HuggingFace, Numpy, etc)
pip install ninja numpy huggingface_hub

# 🧨 O GRANDE MOMENTO: Compila o Kernel CUDA
# O pip -e . vai ler nosso setup.py e o NVCC vai mastigar o C++
pip install -e .
```

*Se chover letras verdes dizendo `Successfully installed socket_engine_cuda`, você é um Deus da infraestrutura.*

---

## 6. O Abastecimento (HuggingFace)
Agora baixamos os 40GB de pesos com integridade garantida.

```bash
# Dê permissão de execução militar ao nosso script Bash
chmod +x setup_weights.sh

# Rode o Manifesto
./setup_weights.sh
```

---

## 7. O Teste de Ignição (Smoke Test)
A hora da verdade. O teste que desenhamos pra confirmar se as placas tectônicas da memória estão no lugar.

```bash
python titan_tester.py
```
Se o script devolver **"A MAQUINA RESPIRA! 🚀"**, você tem em mãos um dos servidores de IA Generativa de mundos 3D mais bem configurados do planeta.
