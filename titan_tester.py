import torch
import time
import logging
from core.monster_pool import TitanVRAMManager

# Import do Kernel C++ Nativo via PyBind11
try:
    import socket_engine_cuda
except ImportError:
    logging.warning("[Titan Tester] Módulo C++ nativo (socket_engine_cuda) não encontrado! Você compilou com 'pip install -e .' ? O teste usará modo simulação se falhar.")
    socket_engine_cuda = None

# =============================================================================
# [TESTER] AWE V6 Titan - The Smoke Test
# O Teste de Respirar: Testa se a TitanBridge aguenta a matemática da VRAM
# antes de fritar 40GB buscando modelos HuggingFace reais.
# =============================================================================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [TitanTester] %(levelname)s - %(message)s')
logger = logging.getLogger("SmokeTest")

def check_vram_density_limit(num_vertices: int, available_vram_bytes: int):
    """
    A Escala do Sucesso:
    D = (V_count * sizeof(float3)) / Memory_available
    Garante que a malha enviada para o Kernel C++ não vai causar Out-Of-Memory na TitanBridge.
    """
    # 1 Vértice Titan = 3(Pos) + 3(Norm) + 2(UV) + 1(ID) = 9 floats * 4 bytes = 36 bytes.
    bytes_per_vertex = 36 
    required_memory = num_vertices * bytes_per_vertex
    
    density_ratio = required_memory / available_vram_bytes
    
    logger.info(f"Densidade Calculada (D): {density_ratio:.4f} ({required_memory / (1024**2):.2f} MB exigidos)")
    
    if density_ratio > 0.85: # Limite de segurança rigoroso (85%)
        raise MemoryError("Densidade Crítica (D > 0.85). O Kernel C++ pode causar colapso (OOM). Reduza Target Poly Count!")
    
    return True

def run_smoke_test():
    logger.info("==================================================")
    logger.info("V6 Titan - INICIANDO SMOKE TEST (C++ IPC + VRAM)")
    logger.info("==================================================")
    
    manager = TitanVRAMManager()
    
    # Simula carga no Slot A (Hunyuan3D)
    manager.acquire_forge_context("Mock_Escultor_Slot_A", lambda: "Loaded")
    
    # -------------------------------------------------------------------------
    # TESTE DA PONTE: Python -> GPU VRAM -> C++ PyBind11 -> Python
    # -------------------------------------------------------------------------
    # Gerando uma malha gigantesca na VRAM (5 Milhões de vértices) simulando a saída do Hunyuan
    num_vertices = 5_000_000
    map_res = 4096 # 4K Mask & Depth
    
    logger.info(f"==> Alocando Malha Mockada ({num_vertices} Verts) e Mapas 4K na VRAM L40S...")
    
    if not torch.cuda.is_available():
        logger.error("CUDA INACESSÍVEL! O PyTorch não detectou a L40S. Saindo.")
        return
        
    # Validando o Teorema de Densidade
    available_vram = torch.cuda.mem_get_info()[0] # [0] is free, [1] is total
    check_vram_density_limit(num_vertices, available_vram)
    
    # As Alocações!
    try:
        t_pos    = torch.randn((num_vertices, 3), dtype=torch.float32, device="cuda")
        t_norm   = torch.ones((num_vertices, 3), dtype=torch.float32, device="cuda")
        t_uv     = torch.rand((num_vertices, 2), dtype=torch.float32, device="cuda")
        t_ids    = torch.full((num_vertices, 1), -1, dtype=torch.int32, device="cuda")
        
        t_depth  = torch.rand((map_res, map_res), dtype=torch.float32, device="cuda") * 50.0 # 50 Metros
        t_mask   = torch.full((map_res, map_res), -1, dtype=torch.int32, device="cuda")
        t_dist   = torch.ones((map_res, map_res), dtype=torch.float32, device="cuda") * 100.0
        
        # Teste Rápido de "Socketing" visualizável: Força um prédio (ID 42) no meio do Mapa 4K
        t_mask[2000:2100, 2000:2100] = 42
        t_dist[2000:2100, 2000:2100] = 10.0 # Perto da Borda
        
        logger.info("[Mock VRAM] Dados inseridos com sucesso sem OOM.")
        
    except RuntimeError as e:
        logger.error(f"FALHA GPU OOM NA ALOCAÇÃO: {e}")
        return

    # Invocação do Kernel C++ Zero-Copy
    if socket_engine_cuda is not None:
        logger.info("==> Invocando Motor C++ PyBind11 (Zero-Copy Pass)...")
        start_time = time.perf_counter()
        
        socket_engine_cuda.forward(
            t_pos, t_norm, t_uv, t_ids, t_depth, t_mask, t_dist, 1.0
        )
        
        torch.cuda.synchronize()
        exec_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"[C++ KERNEL] Músculo de Geometria Executado em {exec_time:.2f} ms!")
        
        # Validação Rápida Matemática
        amostra_id = t_ids[0].item()
        logger.info(f"O C++ alterou o ID da Malha para teste de volta ao Python? Resultado: {amostra_id}")
    else:
        logger.warning("Ponte C++ Desativada. Pulando injeção do Tensor.")
        
    logger.info("==> Teste de Fumaça Concluído. Disparando Iron Flush...")
    manager.iron_flush()
    logger.info("A MAQUINA RESPIRA! 🚀")

if __name__ == "__main__":
    run_smoke_test()
