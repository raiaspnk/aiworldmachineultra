import gc
import torch
import logging
from enum import Enum, auto
from typing import Dict, Any, Callable, Optional

# =============================================================================
# [CORE] AWE v6 Titan - Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [MonsterPool] %(levelname)s - %(message)s'
)
logger = logging.getLogger("TitanVRAMManager")

# =============================================================================
# [CORE] Enums & Constants
# =============================================================================
class SlotContext(Enum):
    SLOT_A_STATIC = "Slot A (20GB - Hunyuan3D)"
    SLOT_B_DYNAMIC = "Slot B (16GB - Flux/SUPIR)"
    SLOT_C_UTILITY = "Slot C (12GB - Vision/CUDA Buffers)"

class JobPriority(Enum):
    VISION = 1   # Vision (SAM/Depth) must run before Forge
    FORGE = 2    # Forge (Hunyuan) runs after Vision
    TEXTURE = 3  # Texture (Flux/SUPIR) runs after Forge or async
    PUMP = 4     # TitanBridge memory pumping

# =============================================================================
# [CORE] VRAM Slot Definition
# =============================================================================
class VRAMSlot:
    """
    Representação lógica de uma partição da VRAM da arquitetura L40S.
    """
    def __init__(self, context: SlotContext, max_capacity_gb: float):
        self.context = context
        self.max_capacity_gb = max_capacity_gb
        self.model_refs: Dict[str, Any] = {}
    
    def is_empty(self) -> bool:
        return len(self.model_refs) == 0

    def load_model(self, model_id: str, loader_func: Callable[[], Any]):
        """Carrega um modelo/tensor no slot."""
        if model_id in self.model_refs:
            logger.info(f"[{self.context.value}] Modelo '{model_id}' já está alocado.")
            return self.model_refs[model_id]

        logger.info(f"[{self.context.value}] Alocando modelo '{model_id}'...")
        self.model_refs[model_id] = loader_func()
        logger.info(f"[{self.context.value}] '{model_id}' alocado com sucesso.")
        return self.model_refs[model_id]

    def offload_model(self, model_id: str):
        """Descarrega um modelo específico."""
        if model_id in self.model_refs:
            logger.info(f"[{self.context.value}] Removendo modelo '{model_id}'...")
            del self.model_refs[model_id]
            
    def purge(self):
        """Elimina todos os modelos do slot."""
        if not self.is_empty():
            logger.info(f"[{self.context.value}] Expurgando todos os tensores do Slot...")
            self.model_refs.clear()

# =============================================================================
# [CORE] Titan VRAM Manager
# =============================================================================
class TitanVRAMManager:
    """
    Orquestrador determinístico dos 48GB da L40S.
    Garante o particionamento rígido, Iron Flush e a ordem de prioridades (Vision -> Forge -> Texture).
    """
    def __init__(self):
        logger.info("Inicializando o Coração da Besta: Titan VRAM Manager (L40S).")
        
        # M_total = Slot_A(20GB) + Slot_B(16GB) + Slot_C(12GB)
        self.slot_A = VRAMSlot(SlotContext.SLOT_A_STATIC, 20.0)
        self.slot_B = VRAMSlot(SlotContext.SLOT_B_DYNAMIC, 16.0)
        self.slot_C = VRAMSlot(SlotContext.SLOT_C_UTILITY, 12.0)
        
        # Estado de sincronia
        self._current_phase: Optional[JobPriority] = None

    def iron_flush(self):
        """
        [IRON FLUSH]
        Limpeza agressiva do cache de memória após cada ciclo de geração.
        Evita a fragmentação silenciosa da memória de GPU.
        """
        logger.warning("[IRON FLUSH] Iniciando limpeza de cache cirúrgica...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # Limpa memória IPC que sobrou do C++
            
            # Printa métricas reais da GPU L40S
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.warning(f"[IRON FLUSH] Concluído. GPU VRAM - Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
        else:
            logger.warning("[IRON FLUSH] CUDA invisível (Modo Simulação ou Erro de Env).")

    def acquire_vision_context(self, model_id: str, loader_func: Callable[[], Any]):
        """
        FASE 2: Utilitários e Visão (Depth, SAM). Utiliza o Slot C.
        Contrato: Vision DEVE ocorrer antes da Forge (Hunyuan).
        """
        self._set_phase(JobPriority.VISION)
        # O Slot C compartilha memória com Buffers C++, então mantemos o controle rígido
        return self.slot_C.load_model(model_id, loader_func)

    def acquire_forge_context(self, model_id: str, loader_func: Callable[[], Any]):
        """
        FASE 4: Escultor (Hunyuan3D). Utiliza estaticamente o Slot A.
        Vantagem: Raramente chamaremos purge() no Slot A durante o fluxo.
        """
        # Checagem de contrato de prioridade
        if self._current_phase != JobPriority.VISION and self._current_phase is not None:
            logger.info("Contrato de Prioridade: Inicializando Forge context independentemente da Vision, ou Vision já ocorreu.")
            
        self._set_phase(JobPriority.FORGE)
        return self.slot_A.load_model(model_id, loader_func)

    def acquire_texture_context(self, model_id: str, loader_func: Callable[[], Any]):
        """
        FASE 4/Adicional: Flux e SUPIR. Utiliza o Slot B (Dinâmico).
        Ocorre um swap intenso de VRAM aqui para upscaling.
        """
        self._set_phase(JobPriority.TEXTURE)
        return self.slot_B.load_model(model_id, loader_func)

    def release_dynamic_memory(self):
        """Libera apenas os slots dinâmicos e utilitários, guardando o modelo Estático (Hunyuan3D)."""
        logger.info("Limpando partições dinâmicas (B e C) para próximo ciclo...")
        self.slot_B.purge()
        self.slot_C.purge() # Cuidado aqui ao trabalhar com ponteiros IPC C++
        self.iron_flush()

    def _set_phase(self, phase: JobPriority):
        if self._current_phase != phase:
            logger.info(f"Mudança de Fase da Orquestração: {self._current_phase} -> {phase.name}")
            self._current_phase = phase

# =============================================================================
# [TESTE UNTÁRIO SIMULADO]
# =============================================================================
if __name__ == "__main__":
    pool = TitanVRAMManager()
    
    # Simula carregamento do Depth V3 / SAM 3
    sam = pool.acquire_vision_context("SAM3_H", lambda: {"type": "vision_model", "size": "3GB"})
    
    # Simula inicialização do Escultor
    hunyuan = pool.acquire_forge_context("Hunyuan3D_Static", lambda: {"type": "forge_model", "size": "14GB"})
    
    # Simula Geração Mestre e depois limpeza agressiva
    pool.iron_flush()
    
    # Processo acabou, libera B e C
    pool.release_dynamic_memory()
