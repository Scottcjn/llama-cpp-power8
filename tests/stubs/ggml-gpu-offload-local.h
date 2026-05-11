#ifndef TEST_GGML_GPU_OFFLOAD_LOCAL_H
#define TEST_GGML_GPU_OFFLOAD_LOCAL_H

int gpu_local_matmul_f32(const float *A, const float *B, float *C, int M, int N, int K);

#endif /* TEST_GGML_GPU_OFFLOAD_LOCAL_H */
