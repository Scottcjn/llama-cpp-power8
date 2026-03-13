/*
 * ggml-mass-integration.h - IBM MASS vector math integration for POWER8
 * Elyan Labs 2025
 *
 * MASS provides optimized implementations of:
 * - vsexp: vectorized exp() for softmax
 * - vstanh: vectorized tanh() for activations
 * - vssqrt: vectorized sqrt() for normalization
 */
#ifndef GGML_MASS_INTEGRATION_H
#define GGML_MASS_INTEGRATION_H

#ifdef GGML_USE_IBM_MASS

#include <massv.h>

/* Batch exp for softmax - uses MASS vsexp */
static inline void ggml_mass_exp_f32(float* dst, const float* src, int n) {
    vsexp(dst, (float*)src, &n);
}

/* Batch tanh for activations - uses MASS vstanh */
static inline void ggml_mass_tanh_f32(float* dst, const float* src, int n) {
    vstanh(dst, (float*)src, &n);
}

/* Batch sqrt for normalization - uses MASS vssqrt */
static inline void ggml_mass_sqrt_f32(float* dst, const float* src, int n) {
    vssqrt(dst, (float*)src, &n);
}

/* Batch log for cross-entropy - uses MASS vslog */
static inline void ggml_mass_log_f32(float* dst, const float* src, int n) {
    vslog(dst, (float*)src, &n);
}

#define GGML_MASS_AVAILABLE 1

#else /* !GGML_USE_IBM_MASS */

#define GGML_MASS_AVAILABLE 0

#endif /* GGML_USE_IBM_MASS */

#endif /* GGML_MASS_INTEGRATION_H */
