/*
 * GPU Matmul Offload Integration for POWER8 + RX 580
 * Elyan Labs - Hook into ggml matmul path
 */

#ifndef GGML_GPU_OFFLOAD_INTEGRATION_H
#define GGML_GPU_OFFLOAD_INTEGRATION_H

#include "ggml-gpu-offload-local.h"
#include <stdbool.h>

// Minimum dimensions for GPU offload (smaller = CPU is faster due to overhead)
#define GPU_OFFLOAD_MIN_M 512
#define GPU_OFFLOAD_MIN_N 512  
#define GPU_OFFLOAD_MIN_K 512

// Try GPU offload for F32 matmul, returns true if handled
static inline bool try_gpu_offload_matmul(
    const float *A, const float *B, float *C,
    int64_t M, int64_t N, int64_t K
) {
    // Skip small matrices
    if (M < GPU_OFFLOAD_MIN_M && N < GPU_OFFLOAD_MIN_N && K < GPU_OFFLOAD_MIN_K) {
        return false;
    }
    
    // Try GPU offload
    if (gpu_local_matmul_f32(A, B, C, (int)M, (int)N, (int)K) == 0) {
        return true;  // GPU handled it
    }
    
    return false;  // Fall back to CPU
}

// Check if tensor dimensions qualify for GPU offload
static inline bool should_try_gpu_offload(int64_t ne0, int64_t ne1, int64_t ne00) {
    // ne0 = output rows (M), ne1 = output cols (N), ne00 = inner dim (K)
    return (ne0 >= GPU_OFFLOAD_MIN_M || ne1 >= GPU_OFFLOAD_MIN_N || ne00 >= GPU_OFFLOAD_MIN_K);
}

#endif // GGML_GPU_OFFLOAD_INTEGRATION_H
