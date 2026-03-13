/*
 * ggml-sparse-softmax.h - Block-Level Sparse Softmax
 * Elyan Labs 2026
 *
 * Uses block-level pruning to minimize branch overhead.
 * Skips entire blocks of 128 values if their max is below threshold.
 */

#ifndef GGML_SPARSE_SOFTMAX_H
#define GGML_SPARSE_SOFTMAX_H

#include <math.h>
#include <string.h>
#include <stdio.h>

/* Only apply to large attention windows */
#define SPARSE_SOFTMAX_MIN_SIZE 1024

/* Block size for pruning - process 128 values at a time */
#define SPARSE_BLOCK_SIZE 128

/* Skip blocks where local max is this far below global max */
#define SPARSE_THRESHOLD 8.0f  /* exp(-8) = 0.00034 - stricter for block-level */

static uint64_t g_sp_calls = 0;
static uint64_t g_sp_blocks_processed = 0;
static uint64_t g_sp_blocks_skipped = 0;

static inline float ggml_vec_soft_max_sparse_f32(
    int n,
    float * __restrict__ dst,
    const float * __restrict__ src,
    float max
) {
    const float cutoff = max - SPARSE_THRESHOLD;
    float sum = 0.0f;
    int i = 0;

    g_sp_calls++;

    /* Process in blocks of SPARSE_BLOCK_SIZE */
    for (; i + SPARSE_BLOCK_SIZE <= n; i += SPARSE_BLOCK_SIZE) {
        /* Find local max for this block */
        float block_max = src[i];
        for (int j = 1; j < SPARSE_BLOCK_SIZE; j++) {
            if (src[i + j] > block_max) block_max = src[i + j];
        }

        /* Skip entire block if local max is below threshold */
        if (block_max < cutoff) {
            memset(dst + i, 0, SPARSE_BLOCK_SIZE * sizeof(float));
            g_sp_blocks_skipped++;
        } else {
            /* Process block normally - no per-element branches */
            for (int j = 0; j < SPARSE_BLOCK_SIZE; j++) {
                float val = expf(src[i + j] - max);
                dst[i + j] = val;
                sum += val;
            }
            g_sp_blocks_processed++;
        }
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        float val = expf(src[i] - max);
        dst[i] = val;
        sum += val;
    }

    return sum;
}

__attribute__((destructor))
static void sparse_report(void) {
    uint64_t total = g_sp_blocks_processed + g_sp_blocks_skipped;
    if (total > 0) {
        fprintf(stderr, "\n=== Block-Level Sparse Softmax ===\n");
        fprintf(stderr, "Calls: %lu | Blocks: %lu processed, %lu skipped (%.1f%%)\n",
                (unsigned long)g_sp_calls,
                (unsigned long)g_sp_blocks_processed,
                (unsigned long)g_sp_blocks_skipped,
                (double)g_sp_blocks_skipped * 100.0 / (double)total);
    }
}

#endif
