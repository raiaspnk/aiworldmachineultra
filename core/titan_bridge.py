import mmap
import ctypes
import os
import struct
import torch
import numpy as np
import logging
from typing import Optional

try:
    import safetensors.numpy as st
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# =============================================================================
# [CORE] AWE V10 Surgery - TitanBridge (Zero-Copy Middleware)
# =============================================================================
# FIX #17: GPUDirect mmap hint (hugepages + MAP_POPULATE)
# FIX #27: Safetensor alignment validation
# =============================================================================
logger = logging.getLogger("TitanBridge")

class TitanBridge:
    """
    O TitanBridge resolve o maior problema de performance entre Python e C++:
    A cópia de dados (Serialization / Pickling).
    
    Usando Arquivos Mapeados em Memória (MMAP) e CUDA IPC (Inter-Process Communication),
    o Python escreve os tensores diretamente em uma área de RAM C/CUDA bruta.
    O C++ lê os ponteiros instantaneamente, sem travar o GIL do Python.
    """
    
    def __init__(self, buffer_name: str = "AWE_Titan_Buffer"):
        self.buffer_name = buffer_name
        self.buffer_size = 0
        self.mm: Optional[mmap.mmap] = None
        self.fd: int = -1
        self.fallback_tensor: Optional[torch.Tensor] = None # V8 Enterprise: Fallback
        logger.info(f"TitanBridge Inicializado. Mapeando ponteiro fantasma: {self.buffer_name}")

    def allocate_shared_ram(self, size_bytes: int):
        """
        Aloca um bloco de memória RAM pura compartilhada via MMAP (Memory Mapped File).
        Útil para trocar Arrays Numpy gigantes (como o mapa de profundidade ou máscaras do SAM).
        """
        if self.mm is not None:
            self.free_shared_ram()
            
        logger.info(f"[TitanBridge] Alocando {size_bytes / (1024**2):.2f} MB de Shared RAM...")
        self.buffer_size = size_bytes
        
        try:
            if os.name == 'nt':
                self.mm = mmap.mmap(-1, self.buffer_size, tagname=self.buffer_name)
            else:
                # FIX #17: GPUDirect hint via hugepages + MAP_POPULATE
                self.fd = os.open(f"/tmp/{self.buffer_name}", os.O_CREAT | os.O_TRUNC | os.O_RDWR)
                os.write(self.fd, b'\x00' * self.buffer_size)
                
                # MAP_POPULATE pre-faults all pages (elimina page-fault latency no primeiro acesso)
                map_flags = mmap.MAP_SHARED
                try:
                    map_flags |= getattr(mmap, 'MAP_POPULATE', 0x8000)
                except Exception:
                    pass
                
                self.mm = mmap.mmap(self.fd, self.buffer_size, map_flags, mmap.PROT_WRITE | mmap.PROT_READ)
                
                # FIX #17: Advise kernel sobre acesso sequencial
                try:
                    import ctypes.util
                    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
                    MADV_HUGEPAGE = 14
                    MADV_SEQUENTIAL = 2
                    mm_buf = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.mm)))
                    libc.madvise(mm_buf, self.buffer_size, MADV_HUGEPAGE)
                    libc.madvise(mm_buf, self.buffer_size, MADV_SEQUENTIAL)
                    logger.info("[FIX #17] MMAP hugepages + MAP_POPULATE ativado (GPUDirect hint).")
                except Exception:
                    logger.debug("[FIX #17] madvise não suportado; performance normal.")
                
            logger.info("[TitanBridge] Shared RAM ativa e pronta para leitura/escrita C++.")
        
        except (PermissionError, OSError) as e:
            logger.warning(f"[TitanBridge] OS File Mmap falhou (Docker permissions?). Fallback para PyTorch Pin Memory. Erro: {e}")
            self.fallback_tensor = torch.empty(self.buffer_size, dtype=torch.uint8, pin_memory=True)
            self.mm = None
            self.fd = -1
            logger.info("[TitanBridge] PyTorch Pin Memory Alocada. Simulando IPC via DMA.")

    def write_tensor_to_ram(self, tensor_or_array, offset: int = 0):
        """
        Escreve um Tensor PyTorch da CPU ou um Numpy Array diretamente na ponte Mapped Memory.
        """

        if self.mm is None and self.fallback_tensor is None:
            raise RuntimeError("TitanBridge Buffer não alocado. Rode allocate_shared_ram primeiro.")
            
        # Converte torch CPU para numpy de forma Zero-Copy view
        if isinstance(tensor_or_array, torch.Tensor):
            if tensor_or_array.is_cuda:
                logger.warning("[TitanBridge] Tensor está na VRAM. Faça .cpu().numpy() antes, ou use IPC Handle.")
                tensor_or_array = tensor_or_array.detach().cpu().numpy()
            else:
                tensor_or_array = tensor_or_array.detach().numpy()
                
        # Escreve os bytes puros na memória C-contiguous
        raw_bytes = tensor_or_array.tobytes()
        bytes_len = len(raw_bytes)
        
        if offset + bytes_len > self.buffer_size:
            raise ValueError("Buffer Overflow na TitanBridge! O Tensor excede a RAM alocada.")
            
        # V8 Enterprise: Escreve no MMAP Mágico ou no Fallback DMA
        if self.mm is not None:
            self.mm.seek(offset)
            self.mm.write(raw_bytes)
        elif self.fallback_tensor is not None:
            # Copiar os bytes para o tensor de fallback em Pin Memory
            byte_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
            self.fallback_tensor[offset:offset+bytes_len].copy_(byte_tensor)
            
        logger.info(f"[TitanBridge] {bytes_len / (1024**2):.2f} MB escritos na Offset {offset}.")

    def share_cuda_ipc_handle(self, cuda_tensor: torch.Tensor):
        """
        [THE HOLY GRAIL] 
        Compartilha um Ponteiro IPC (Inter-Process Communication) nativo de CUDA.
        Isso permite que o Python passe uma malha de 10 milhões de polígonos (Hunyuan) 
        para o C++ (SocketEngine) SEM TIRAR OS DADOS DA GPU L40S.
        """
        if not cuda_tensor.is_cuda:
            raise ValueError("[TitanBridge] O Tensor precisa estar na VRAM para IPC Sharing.")
            
        # torch.multiprocessing serve para Python-Python, mas no C++ do backend usamos 
        # a serialização do storage direto. (PyTorch C++ Backend C10).
        try:
            storage = cuda_tensor.untyped_storage()
            ipc_handle = storage._share_cuda_()
            logger.info("[TitanBridge] CUDA IPC Handle gerado. C++ SocketEngine pode acessar VRAM nativa.")
            return ipc_handle
        except Exception as e:
            # V8 Enterprise: Fallback IPC 
            logger.warning(f"[TitanBridge] IPC Handle não suportado no kernel Linux atual: {e}. Desligamento gracioso de zero-copy e retornando ponteiro local.")
            return cuda_tensor.data_ptr()

    def free_shared_ram(self):
        """Libera os ponteiros do sistema."""
        if self.mm is not None:
            logger.info("[TitanBridge] Fechando Mapped Memory e limpando ponteiros...")
            self.mm.close()
            self.mm = None
            if self.fd != -1:
                os.close(self.fd)
                self.fd = -1
        
        if self.fallback_tensor is not None:
            del self.fallback_tensor
            self.fallback_tensor = None
    
    # =========================================================================
    # FIX #27: SAFETENSOR ALIGNMENT VALIDATION
    # =========================================================================
    @staticmethod
    def validate_safetensor_alignment(filepath: str) -> bool:
        """
        FIX #27: safetensors pre-header != 8-byte aligned → mmap reads garbage.
        Valida alinhamento antes de carregar.
        """
        if not os.path.exists(filepath):
            logger.error(f"[FIX #27] Arquivo não encontrado: {filepath}")
            return False
        
        file_size = os.path.getsize(filepath)
        if file_size < 8:
            logger.error(f"[FIX #27] Arquivo muito pequeno: {file_size} bytes")
            return False
        
        with open(filepath, 'rb') as f:
            # safetensors header: 8 bytes LE uint64 = tamanho do JSON header
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            
            # Verifica alinhamento do header
            data_offset = 8 + header_size
            if data_offset % 8 != 0:
                logger.warning(f"[FIX #27] safetensor data offset {data_offset} não é 8-byte aligned! "
                               f"Header size: {header_size}. mmap pode ler garbage.")
                return False
            
            if data_offset > file_size:
                logger.error(f"[FIX #27] Header size ({header_size}) maior que arquivo ({file_size})!")
                return False
        
        logger.info(f"[FIX #27] safetensor '{filepath}' validado: header={header_size}B, "  
                     f"data_offset={data_offset}B (aligned).")
        return True
    
    @staticmethod
    def safe_load_safetensor(filepath: str) -> dict:
        """
        FIX #27: Wrapper seguro para safetensors.load_file() com validação.
        """
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors não instalado!")
        
        if not TitanBridge.validate_safetensor_alignment(filepath):
            logger.warning(f"[FIX #27] Alignment check falhou. Usando load sem mmap.")
            # Fallback: load sem mmap (mais lento mas seguro)
            with open(filepath, 'rb') as f:
                data = f.read()
            from safetensors.numpy import load as st_load_bytes
            return st_load_bytes(data)
        
        return st.load_file(filepath)
                
# =============================================================================
# [TESTE UNITÁRIO SIMULADO]
# =============================================================================
if __name__ == "__main__":
    # Simula 100MB de pontos 3D
    simulated_point_cloud = np.ones((1000000, 3), dtype=np.float32)
    bytes_needed = simulated_point_cloud.nbytes
    
    bridge = TitanBridge(buffer_name="L40S_Geo_Pipeline")
    bridge.allocate_shared_ram(bytes_needed)
    bridge.write_tensor_to_ram(simulated_point_cloud)
    
    # O C++ nesse momento estaria lendo do outro lado...
    bridge.free_shared_ram()
