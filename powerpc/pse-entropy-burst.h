/*
 * pse-entropy-burst.h - Incremental Burst PSE Entropy for llama.cpp
 *
 * Optimized entropy injection that maintains VSX speed:
 * - Apply entropy every N tokens (not every token)
 * - Only perturb top-K candidates (not full 32K vocab)
 * - Stronger bursts compensate for lower frequency
 *
 * Result: Same behavioral divergence, minimal performance impact.
 */

#ifndef PSE_ENTROPY_BURST_H
#define PSE_ENTROPY_BURST_H

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C99 bool support */
#ifndef __cplusplus
#include <stdbool.h>
#endif

/*===========================================================================
 * Configuration
 *===========================================================================*/

/* Burst interval: Apply entropy every N tokens (1 = every token, 4 = every 4th) */
#ifndef PSE_BURST_INTERVAL
#define PSE_BURST_INTERVAL 4
#endif

/* Burst strength: Multiplied by interval to maintain total effect */
#ifndef PSE_BURST_STRENGTH
#define PSE_BURST_STRENGTH 0.08f  /* 4x normal (0.02) for interval=4 */
#endif

/* Top-K for entropy: Only perturb top candidates (0 = all) */
#ifndef PSE_TOPK_ENTROPY
#define PSE_TOPK_ENTROPY 512  /* Top 512 candidates get entropy */
#endif

/* Enable collapse resonance points (vec_perm activation) */
#ifndef PSE_COLLAPSE_ENABLED
#define PSE_COLLAPSE_ENABLED 1
#endif

/* Collapse interval: Heavy collapse every N tokens */
#ifndef PSE_COLLAPSE_INTERVAL
#define PSE_COLLAPSE_INTERVAL 16
#endif

/* Golden ratio for mixing */
#define PSE_PHI 0x9E3779B9U

/*===========================================================================
 * State Tracking
 *===========================================================================*/

static int64_t g_pse_token_pos = 0;
static uint64_t g_pse_bursts = 0;
static uint64_t g_pse_collapses = 0;
static bool g_pse_banner_shown = false;

/*===========================================================================
 * Hardware Timebase
 *===========================================================================*/

static inline uint64_t pse_read_timebase(void) {
#if defined(__powerpc64__) || defined(__powerpc__)
    uint64_t tb;
    __asm__ __volatile__("mftb %0" : "=r"(tb));
    return tb;
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return (uint64_t)&g_pse_token_pos;
#endif
}

/*===========================================================================
 * PSE Banner Display
 *===========================================================================*/

static inline void pse_show_banner(void) {
    if (!g_pse_banner_shown) {
        g_pse_banner_shown = true;
        fprintf(stderr, "\n");
        fprintf(stderr, "PSE Burst Entropy Active\n");
        fprintf(stderr, "  Interval: %d | Strength: %.3f | TopK: %d\n",
                PSE_BURST_INTERVAL, PSE_BURST_STRENGTH, PSE_TOPK_ENTROPY);
        fprintf(stderr, "  Platform: %s\n",
#if defined(__powerpc64__)
                "POWER8 (mftb timebase)"
#elif defined(__x86_64__)
                "x86_64 (rdtsc)"
#elif defined(__aarch64__)
                "ARM64 (cntvct)"
#else
                "Generic"
#endif
        );
        fprintf(stderr, "\n");
    }
}

/*===========================================================================
 * Pure C entropy functions (for ggml C code)
 *===========================================================================*/

/* Apply entropy burst to raw float logits array */
static inline void pse_apply_entropy_burst_float(float* logits, size_t n_vocab) {
    g_pse_token_pos++;
    pse_show_banner();

    /* Skip non-burst tokens for speed */
    if (g_pse_token_pos % PSE_BURST_INTERVAL != 0) {
        return;
    }

    if (!logits || n_vocab == 0) return;

    g_pse_bursts++;

    uint64_t tb = pse_read_timebase();
    uint32_t base_seed = (uint32_t)(tb ^ (tb >> 32));
    base_seed ^= (uint32_t)g_pse_token_pos * PSE_PHI;

    /* Determine how many candidates to affect */
    size_t entropy_count = (PSE_TOPK_ENTROPY > 0 && (size_t)PSE_TOPK_ENTROPY < n_vocab)
                           ? (size_t)PSE_TOPK_ENTROPY : n_vocab;

    /* Apply burst entropy to top candidates */
    for (size_t i = 0; i < entropy_count; i++) {
        uint32_t seed = base_seed ^ ((uint32_t)i * PSE_PHI);
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        float entropy = ((float)(seed & 0xFFFFU) / 65536.0f) - 0.5f;
        logits[i] += entropy * PSE_BURST_STRENGTH;
    }

#if PSE_COLLAPSE_ENABLED
    /* Heavy collapse at resonance points */
    if (g_pse_token_pos % PSE_COLLAPSE_INTERVAL == 0) {
        /* Find mean logit for hot/cold threshold */
        float sum = 0.0f;
        for (size_t i = 0; i < entropy_count; i++) {
            sum += logits[i];
        }
        float mean = sum / (float)entropy_count;

        /* Entropy-seeded threshold adjustment */
        float threshold_adjust = ((float)(base_seed & 0xFF) / 256.0f - 0.5f) * 0.1f;
        float threshold = mean + threshold_adjust;

        /* Collapse: Amplify hot, dampen cold (non-bijunctive) */
        for (size_t i = 0; i < entropy_count; i++) {
            if (logits[i] > threshold) {
                logits[i] *= 1.05f;  /* Hot path: Amplify */
            } else {
                logits[i] *= 0.95f;  /* Cold path: Dampen */
            }
        }
        g_pse_collapses++;
    }
#endif
}

/*===========================================================================
 * Reset and Metrics
 *===========================================================================*/

static inline void pse_reset(void) {
    g_pse_token_pos = 0;
    g_pse_bursts = 0;
    g_pse_collapses = 0;
}

static inline void pse_report_metrics(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "PSE Burst Entropy Metrics\n");
    fprintf(stderr, "  Tokens processed: %ld\n", (long)g_pse_token_pos);
    fprintf(stderr, "  Entropy bursts:   %lu (every %d tokens)\n",
            (unsigned long)g_pse_bursts, PSE_BURST_INTERVAL);
    fprintf(stderr, "  Collapse events:  %lu (every %d tokens)\n",
            (unsigned long)g_pse_collapses, PSE_COLLAPSE_INTERVAL);
    fprintf(stderr, "  Burst strength:   %.4f\n", PSE_BURST_STRENGTH);
    fprintf(stderr, "  TopK affected:    %d\n", PSE_TOPK_ENTROPY);
}

#ifdef __cplusplus
} /* extern "C" */

/*===========================================================================
 * C++ Template versions (for llama.cpp token_data structs)
 *===========================================================================*/

/*
 * Apply burst entropy to token candidates (llama_token_data array).
 * Only activates every PSE_BURST_INTERVAL tokens.
 * Only affects top PSE_TOPK_ENTROPY candidates.
 */
template<typename T>
static inline void pse_apply_entropy_burst(T* cur, size_t n_vocab) {
    g_pse_token_pos++;
    pse_show_banner();

    /* Skip non-burst tokens for speed */
    if (g_pse_token_pos % PSE_BURST_INTERVAL != 0) {
        return;
    }

    if (!cur || n_vocab == 0) return;

    g_pse_bursts++;

    uint64_t tb = pse_read_timebase();
    uint32_t base_seed = (uint32_t)(tb ^ (tb >> 32));
    base_seed ^= (uint32_t)g_pse_token_pos * PSE_PHI;

    /* Determine how many candidates to affect */
    size_t entropy_count = (PSE_TOPK_ENTROPY > 0 && (size_t)PSE_TOPK_ENTROPY < n_vocab)
                           ? (size_t)PSE_TOPK_ENTROPY : n_vocab;

    /* Apply burst entropy to top candidates */
    for (size_t i = 0; i < entropy_count; i++) {
        uint32_t seed = base_seed ^ ((uint32_t)i * PSE_PHI);
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        float entropy = ((float)(seed & 0xFFFFU) / 65536.0f) - 0.5f;
        cur[i].logit += entropy * PSE_BURST_STRENGTH;
    }

#if PSE_COLLAPSE_ENABLED
    /* Heavy collapse at resonance points */
    if (g_pse_token_pos % PSE_COLLAPSE_INTERVAL == 0) {
        pse_apply_collapse_resonance(cur, entropy_count, base_seed);
        g_pse_collapses++;
    }
#endif
}

#if PSE_COLLAPSE_ENABLED
/*
 * Apply collapse at resonance points.
 * Amplifies "hot" candidates (high logit) and prunes "cold" ones.
 * Non-bijunctive transformation - creates emergent patterns.
 */
template<typename T>
static inline void pse_apply_collapse_resonance(T* cur, size_t count, uint32_t seed) {
    if (count < 8) return;

    /* Find mean logit for hot/cold threshold */
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += cur[i].logit;
    }
    float mean = sum / (float)count;

    /* Entropy-seeded threshold adjustment */
    float threshold_adjust = ((float)(seed & 0xFF) / 256.0f - 0.5f) * 0.1f;
    float threshold = mean + threshold_adjust;

    /* Collapse: Amplify hot, dampen cold (non-bijunctive) */
    for (size_t i = 0; i < count; i++) {
        if (cur[i].logit > threshold) {
            cur[i].logit *= 1.05f;  /* Hot path: Amplify (dup from vec_perm) */
        } else {
            cur[i].logit *= 0.95f;  /* Cold path: Dampen (prune effect) */
        }
    }
}
#endif

/* Compatibility alias */
template<typename T>
static inline void pse_apply_entropy(T* cur, size_t n_vocab) {
    pse_apply_entropy_burst(cur, n_vocab);
}

#endif /* __cplusplus */

#endif /* PSE_ENTROPY_BURST_H */
