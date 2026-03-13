#pragma once

#include <torch/extension.h>

// ============================================================================
// MonsterCore v2 - Pure CUDA Kernels
// ============================================================================

// Launches the RANSAC kernel optimized with warp reductions
torch::Tensor launch_gpu_ransac_hard_surface(
    torch::Tensor vertices,
    float distance_threshold,
    int num_iterations,
    int batch_size
);

// Launches the Taubin Smooth kernel optimized with shared memory (Volume-Preserving)
torch::Tensor launch_gpu_taubin_smooth(
    torch::Tensor vertices,
    torch::Tensor faces,
    int iterations,
    float lambda_factor,
    float mu_factor
);

// Launches the GPU Terrain Generation kernel (Depth Displacement) with Tiling Extents and Smoothing
std::vector<torch::Tensor> launch_gpu_generate_world_geometry(
    torch::Tensor depth_map,
    float max_height,
    float offset_x,
    float offset_y,
    float scale,
    int smooth_iters,
    float smooth_lambda,
    float smooth_mu
);

// V5: World Assembler - Fast concatenation and batch Transformation
std::vector<torch::Tensor> launch_gpu_assemble_world(
    std::vector<torch::Tensor> verts_list,
    std::vector<torch::Tensor> faces_list,
    std::vector<torch::Tensor> transforms_list
);
