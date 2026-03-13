from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# =============================================================================
# [BUILD] AWE V6 Titan - PyBind11 Setup
# O Aperto de Mão C++ / Python
# =============================================================================

# Forçar compilação altamente otimizada para arquitetura Ada Lovelace (L40S)
# Compute Capability da L40S é 8.9
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

setup(
    name='socket_engine_cuda',
    ext_modules=[
        CUDAExtension(
            name='socket_engine_cuda', 
            sources=['kernels/socket_engine.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '--use_fast_math'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='6.0.0',
    description='V6 Titan Zero-Copy Socket Engine',
)
