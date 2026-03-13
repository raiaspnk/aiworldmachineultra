#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

// ============================================================================
// [KERNELS] AWE V10 Surgery - PyBind11 Socket Engine C++ / CUDA
// Zero-Copy Transfer definitivo. Lemos os Tensores diretamente da GDDR6 (VRAM)!
// ============================================================================
// FIX #2:  __launch_bounds__(256) em todos os kernels custom
// FIX #18: blockDim forçado a 256 + grid-stride loop (sem __syncthreads deadlock)
// ============================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " precisa ser um Tensor CUDA!")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " precisa ser C-contiguous!")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// FIX #2: Constante de block size universal (múltiplo de warpSize=32, funciona em L40S, A100, H100, RTX 4090)
constexpr int UNIVERSAL_BLOCK_SIZE = 256;

// ============================================================================
// [CUDA KERNEL] TRUE VECTOR DISPLACEMENT & BLENDING RAMP SOCKETING
// FIX #2: __launch_bounds__ garante que o compilador otimiza registros 
//         para qualquer SM (Ada, Ampere, Hopper) sem estourar IPC.
// ============================================================================
__global__ 
__launch_bounds__(UNIVERSAL_BLOCK_SIZE)
void d_apply_vector_diplacement_and_socketing(
    float* positions,      // [V, 3]
    const float* normals,  // [V, 3]
    const float* uvs,      // [V, 2]
    int* semantic_ids,     // [V, 1]
    int num_vertices, 
    const float* depth_map_4k,        
    const int* sam_mask_4k,           
    const float* distance_field_4k,   
    int map_width, 
    int map_height,
    float height_scale
) {
    // V8/V10: Grid-Stride Loop (Agnostic to hardware warp/block limits)
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x) {
        
        // UVs Base (0 a 1)
        float local_u = uvs[idx * 2 + 0];
        float local_v = uvs[idx * 2 + 1];

        // Projetando UV pro Pixel 4K
        int px_x = (int)(local_u * (map_width - 1));
        int px_y = (int)(local_v * (map_height - 1));
        
        // Clamp de Borda
        px_x = max(0, min(px_x, map_width - 1));
        px_y = max(0, min(px_y, map_height - 1));
        int pixel_idx = (px_y * map_width) + px_x;

        // Leituras Métricas e Semânticas
        float depth_val = depth_map_4k[pixel_idx]; 
        int object_id = sam_mask_4k[pixel_idx];
        float dist_to_edge = distance_field_4k[pixel_idx]; 

        semantic_ids[idx] = object_id;

        // Extraindo Vec3 do Array Flattened
        float px = positions[idx * 3 + 0];
        float py = positions[idx * 3 + 1];
        float pz = positions[idx * 3 + 2];
        
        float nx = normals[idx * 3 + 0];
        float ny = normals[idx * 3 + 1];
        float nz = normals[idx * 3 + 2];

        if (object_id == -1) {
            // [MODO 1: NATUREZA (TRUE VECTOR DISPLACEMENT)]
            float displacement = depth_val * height_scale;
            positions[idx * 3 + 0] = px + nx * displacement;
            positions[idx * 3 + 1] = py + ny * displacement;
            positions[idx * 3 + 2] = pz + nz * displacement;
        } else {
            // [MODO 2: INDUSTRIAL SOCKETING WITH BLEND RAMP]
            float anchor_z = 15.0f; 
            float ramp_width_pixels = 40.0f; 
            
            if (dist_to_edge >= ramp_width_pixels) {
                // Platô Rígido (Core do Prédio)
                positions[idx * 3 + 2] = anchor_z;
            } else {
                // Suavização do Degrau (Rampa Exponencial Suave)
                float alpha = dist_to_edge / ramp_width_pixels; 
                float wild_displacement = depth_val * height_scale;
                float wild_z = pz + (nz * wild_displacement);
                
                float smooth_alpha = alpha * alpha * (3.0f - 2.0f * alpha);
                positions[idx * 3 + 2] = (wild_z * (1.0f - smooth_alpha)) + (anchor_z * smooth_alpha);
                
                positions[idx * 3 + 0] = px + (nx * wild_displacement) * (1.0f - smooth_alpha);
                positions[idx * 3 + 1] = py + (ny * wild_displacement) * (1.0f - smooth_alpha);
            }
        }
    } // End Grid-Stride
}

// ============================================================================
// [CUDA KERNEL 2] CATENARY PHYSICS (The Physics of Sagging Cables)
// FIX #2:  __launch_bounds__ para portabilidade entre SMs
// FIX #18: Grid-stride puro, sem __syncthreads(), sem __shfl_xor_sync()
//          blockDim SEMPRE múltiplo de warpSize → zero race condition
// ============================================================================
__global__ 
__launch_bounds__(UNIVERSAL_BLOCK_SIZE)
void d_simulate_catenary(
    float* out_points,      // [N, C, 3] Output array
    const float* start_pts, // [N, 3]
    const float* end_pts,   // [N, 3]
    float sag,              // Tensão do cabo (a param)
    int points_per_cable,
    int total_cables
) {
    // FIX #18: Grid-Stride puro — cada thread processa 1 cabo independente
    // Sem __syncthreads, sem warp-shuffle, sem divergência de execução
    for (int cable_idx = blockIdx.x * blockDim.x + threadIdx.x; cable_idx < total_cables; cable_idx += blockDim.x * gridDim.x) {

        float sx = start_pts[cable_idx * 3 + 0];
        float sy = start_pts[cable_idx * 3 + 1];
        float sz = start_pts[cable_idx * 3 + 2];

        float ex = end_pts[cable_idx * 3 + 0];
        float ey = end_pts[cable_idx * 3 + 1];
        float ez = end_pts[cable_idx * 3 + 2];

        // Distância linear 2D (Chão)
        float dx = ex - sx;
        float dy = ey - sy;
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 0.001f) dist = 0.001f;

        // Vetores Direcionais
        float dir_x = dx / dist;
        float dir_y = dy / dist;

        // A Catenária real f(x) = a * cosh(x/a) - a
        float a = sag; 
        
        // Offset para focar o sag no meio exato
        float mid_x = dist / 2.0f;

        for(int i = 0; i < points_per_cable; i++) {
            float t = (float)i / (float)(points_per_cable - 1);
            float current_dist = t * dist;
            
            float px = sx + dir_x * current_dist;
            float py = sy + dir_y * current_dist;
            
            float target_z = sz + t * (ez - sz);
            
            float shifted_x = current_dist - mid_x;
            float sag_drop = a * coshf(shifted_x / a) - a;
            
            float final_z = target_z - (sag_drop * 0.5f);

            int write_idx = (cable_idx * points_per_cable + i) * 3;
            out_points[write_idx + 0] = px;
            out_points[write_idx + 1] = py;
            out_points[write_idx + 2] = final_z;
        }
    } // End Grid-Stride
}

// ============================================================================
// [PYBIND11 WRAPPER] A Ponte Direta C++ <-> Python
// FIX #2: Usa UNIVERSAL_BLOCK_SIZE constante ao invés de dynamic occupancy
//         para garantir consistência entre todas as GPUs.
//         cudaOccupancyMaxPotentialBlockSize calibra para matMul genérico
//         e retorna tamanhos sub-ótimos para nossos kernels custom.
// ============================================================================
void socket_engine_forward(
    torch::Tensor positions,
    torch::Tensor normals,
    torch::Tensor uvs,
    torch::Tensor semantic_ids,
    torch::Tensor depth_map,
    torch::Tensor sam_mask,
    torch::Tensor distance_field,
    float height_scale
) {
    CHECK_INPUT(positions);
    CHECK_INPUT(normals);
    CHECK_INPUT(uvs);
    CHECK_INPUT(semantic_ids);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(sam_mask);
    CHECK_INPUT(distance_field);

    int num_vertices = positions.size(0);
    int map_height = depth_map.size(0);
    int map_width = depth_map.size(1);

    // FIX #2: Block size fixo e universal (256 = 8 warps = ótimo para todos SMs)
    const int blockSize = UNIVERSAL_BLOCK_SIZE;
    int gridSize = (num_vertices + blockSize - 1) / blockSize;
    
    // Cap grid para evitar overhead de dispatch em GPUs pequenas
    int maxGrid = 65535;
    gridSize = min(gridSize, maxGrid);

    d_apply_vector_diplacement_and_socketing<<<gridSize, blockSize>>>(
        positions.data_ptr<float>(),
        normals.data_ptr<float>(),
        uvs.data_ptr<float>(),
        semantic_ids.data_ptr<int>(),
        num_vertices,
        depth_map.data_ptr<float>(),
        sam_mask.data_ptr<int>(),
        distance_field.data_ptr<float>(),
        map_width, 
        map_height,
        height_scale
    );
    
    // Sync + error check
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[V10 CUDA Error] SocketEngine falhou: " 
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA Error in socket_engine: ") + cudaGetErrorString(err));
    }
}

// ----------------------------------------------------------------------------
// Wrapper pro Python (Catenary)
// FIX #18: blockDim=256 constante, sem race condition
// ----------------------------------------------------------------------------
void simulate_catenary_forward(
    torch::Tensor out_points,
    torch::Tensor start_pts,
    torch::Tensor end_pts,
    float sag,
    int points_per_cable
) {
    CHECK_INPUT(out_points);
    CHECK_INPUT(start_pts);
    CHECK_INPUT(end_pts);
    
    int total_cables = start_pts.size(0);
    
    // FIX #18: Block size fixo = 256 (múltiplo de warpSize=32)
    // Nunca mais blockDim.x % warpSize != 0
    const int blockSize = UNIVERSAL_BLOCK_SIZE;
    int gridSize = (total_cables + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 65535);

    d_simulate_catenary<<<gridSize, blockSize>>>(
        out_points.data_ptr<float>(),
        start_pts.data_ptr<float>(),
        end_pts.data_ptr<float>(),
        sag,
        points_per_cable,
        total_cables
    );
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[V10 CUDA Error] Catenary Simul falhou: " 
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("CUDA Error in simulate_catenary: ") + cudaGetErrorString(err));
    }
}


// Registro do Módulo Pybind11 para o Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &socket_engine_forward, "V10 Socket Engine Kernel (CUDA) - Universal Block Size");
    m.def("simulate_catenary", &simulate_catenary_forward, "V10: CUDA Catenary Physics - Deadlock Free");
}
