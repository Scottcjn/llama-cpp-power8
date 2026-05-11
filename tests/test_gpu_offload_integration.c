#include <stdbool.h>
#include <stdint.h>

#include "ggml-gpu-offload-integration.h"

static int failures;
static int stub_calls;
static int stub_return_code;
static int captured_m;
static int captured_n;
static int captured_k;

int gpu_local_matmul_f32(const float *A, const float *B, float *C, int M, int N, int K)
{
    (void)A;
    (void)B;
    (void)C;
    stub_calls++;
    captured_m = M;
    captured_n = N;
    captured_k = K;
    return stub_return_code;
}

static void check(bool condition)
{
    if (!condition) {
        failures++;
    }
}

static void reset_stub(int return_code)
{
    stub_calls = 0;
    stub_return_code = return_code;
    captured_m = 0;
    captured_n = 0;
    captured_k = 0;
}

static void test_should_try_gpu_offload_boundaries(void)
{
    check(!should_try_gpu_offload(511, 511, 511));
    check(should_try_gpu_offload(512, 1, 1));
    check(should_try_gpu_offload(1, 512, 1));
    check(should_try_gpu_offload(1, 1, 512));
    check(should_try_gpu_offload(2048, 4, 4));
}

static void test_small_matmul_never_calls_gpu(void)
{
    float A[1] = {1.0f};
    float B[1] = {2.0f};
    float C[1] = {0.0f};

    reset_stub(0);
    check(!try_gpu_offload_matmul(A, B, C, 8, 16, 32));
    check(stub_calls == 0);
}

static void test_large_matmul_reports_gpu_success(void)
{
    float A[1] = {1.0f};
    float B[1] = {2.0f};
    float C[1] = {0.0f};

    reset_stub(0);
    check(try_gpu_offload_matmul(A, B, C, 512, 16, 32));
    check(stub_calls == 1);
    check(captured_m == 512);
    check(captured_n == 16);
    check(captured_k == 32);
}

static void test_large_matmul_falls_back_on_gpu_failure(void)
{
    float A[1] = {1.0f};
    float B[1] = {2.0f};
    float C[1] = {0.0f};

    reset_stub(-1);
    check(!try_gpu_offload_matmul(A, B, C, 64, 64, 1024));
    check(stub_calls == 1);
    check(captured_m == 64);
    check(captured_n == 64);
    check(captured_k == 1024);
}

int main(void)
{
    test_should_try_gpu_offload_boundaries();
    test_small_matmul_never_calls_gpu();
    test_large_matmul_reports_gpu_success();
    test_large_matmul_falls_back_on_gpu_failure();

    return failures == 0 ? 0 : 1;
}
