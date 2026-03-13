#!/bin/bash
# =============================================================================
# [INFRA] AWE V6 Titan - Manifesto dos Monstros (setup_weights.sh)
# Gerenciador Industrial de Pesos Open Weights
# Funcionalidades: Checksum SHA-256 + Symlinks de Volume para evitar cópias
# =============================================================================

# Defina seu Volume Permanente de Pesos aqui (ex: RunPod Network Volume)
GLOBAL_WEIGHTS_DIR="/workspace/models"
LOCAL_WEIGHTS_DIR="./weights"

# Criação do Hub Local
mkdir -p "$LOCAL_WEIGHTS_DIR"

# Função de Log Verde Titan
function t_log {
    echo -e "\e[32m[Manifesto Titan]\e[0m $1"
}

function t_err {
    echo -e "\e[31m[ERRO Crítico]\e[0m $1"
    exit 1
}

# =============================================================================
# Função Core: Sincronização, Hashing e Symlink
# =============================================================================
function sync_monster() {
    local monster_dir=$1
    local file_name=$2
    local hug_repo=$3
    local expected_sha256=$4

    local global_path="$GLOBAL_WEIGHTS_DIR/$monster_dir/$file_name"
    local local_path="$LOCAL_WEIGHTS_DIR/$monster_dir/$file_name"

    mkdir -p "$GLOBAL_WEIGHTS_DIR/$monster_dir"
    mkdir -p "$LOCAL_WEIGHTS_DIR/$monster_dir"

    t_log "Verificando Ativo: $file_name"

    # Passo 1: Download se não existe no Hub Global
    if [ ! -f "$global_path" ]; then
        t_log "-> Baixando do HuggingFace ($hug_repo)..."
        huggingface-cli download "$hug_repo" "$file_name" --local-dir "$GLOBAL_WEIGHTS_DIR/$monster_dir" --local-dir-use-symlinks False
    fi

    # Passo 2: Verificação de Integridade (SHA-256)
    t_log "-> Validando Criptografia SHA-256..."
    actual_sha256=$(sha256sum "$global_path" | awk '{print $1}')
    if [ "$actual_sha256" != "$expected_sha256" ]; then
        t_err "Checksum FALHOU para $file_name! Arquivo corrompido."
    fi
    t_log "-> Checksum Válido [OK]"

    # Passo 3: Criar Symlink no Projeto Local (Zero-Copy HD)
    if [ ! -L "$local_path" ]; then
        ln -s "$global_path" "$local_path"
        t_log "-> Symlink criado em $local_path"
    fi
    echo "---------------------------------------------------"
}

# =============================================================================
# A BÍBLIA DOS MONSTROS (Open Weights Registrados)
# As hashes aqui são mocks para estrutura. Em PRD usaríamos os hashes reais HF.
# =============================================================================
t_log "Iniciando Particionamento Gênesis de Pesos (L40S V6 Titan)..."

# Slot A: The Forge (TRELLIS O-Voxel + Decoders Ocultos)
sync_monster "forge" "trellis_2_ovoxel.safetensors" "microsoft/TRELLIS.2" "mock_hash_trellis2"
sync_monster "forge" "trellis_2_5b_shape_diffuse.safetensors" "microsoft/TRELLIS.2-2.5B" "mock_hash_trellis_25b"
sync_monster "forge" "trellis_decoder_512.pt" "discord_archive/TRELLIS_hidden" "mock_hash_trellis_dec"

# Slot B: Texture God (FLUX.2-dev + QZeros + OpenCLIP + SUPIR)
sync_monster "texture" "flux2-dev-bnb-4bit.safetensors" "black-forest-labs/FLUX.2-dev-bnb-4bit" "mock_hash_flux2"
sync_monster "texture" "qzeros.pt" "black-forest-labs/FLUX.2-dev-bnb-4bit" "mock_hash_qzeros"
sync_monster "texture" "openclip_vit_g14_laion2b.pt" "laion/CLIP-ViT-g-14-laion2B-s34B-b88K" "mock_hash_openclip"
sync_monster "texture" "supir_v0.pt" "SUPIR/Model" "mock_hash_supir"

# Slot C: Vision Lab & Architect & QC
sync_monster "vision" "sam_vit_h_4b8939.pth" "facebook/sam-hq" "mock_hash_sam"
sync_monster "vision" "depth_anything_v3_metric.pt" "LiheYoung/depth-anything-v3" "mock_hash_depth"
sync_monster "vision" "midas_v3_1_small_indoor.pt" "isl-org/MiDaS" "mock_hash_midas"
sync_monster "vision" "mobilenet_v3_small_qc.pt" "google/mobilenet_v3" "mock_hash_mobilenet"

t_log "🔥 TODOS OS MONSTROS ESTÃO PRONTOS NO HOST! PODE DAR BOOT NA ENGINE."
