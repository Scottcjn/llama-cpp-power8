/*
 * ggml-dcbt-resident.h - Full L2/L3 Resident Prefetch for POWER8
 *
 * THE MISSING PIECE: TH=0x10 full block resident keeps weights HOT
 * This is what enabled 117 t/s peak - Claude dropped it in updates!
 *
 * POWER8 dcbt Hint Types:
 * - TH=0x00: Normal prefetch (may evict)
 * - TH=0x10: FULL BLOCK RESIDENT - stays in L2/L3 until explicit invalidate
 * - TH=0x18: Partial resident
 */

#ifndef GGML_DCBT_RESIDENT_H
#define GGML_DCBT_RESIDENT_H

#ifdef __powerpc64__

/*===========================================================================
 * FULL RESIDENT PREFETCH - The 117 t/s Enabler
 *
 * dcbt with TH=0x10 marks cache lines as "resident" - they stay hot
 * in L2 (512KB/core) and L3 (8MB/core) until explicitly evicted.
 *
 * This is CRITICAL for matmul where we reuse weights across many tokens.
 *===========================================================================*/

/* Full block resident - KEEPS WEIGHTS HOT */
#define DCBT_RESIDENT_FULL(addr) \
    __asm__ __volatile__("dcbt 16, %0, 0" : : "b"(addr) : "memory")

/* Partial resident - for collapse patterns */
#define DCBT_RESIDENT_PARTIAL(addr) \
    __asm__ __volatile__("dcbt 24, %0, 0" : : "b"(addr) : "memory")

/* Normal prefetch (for comparison) */
#define DCBT_NORMAL(addr) \
    __asm__ __volatile__("dcbt 0, %0" : : "r"(addr))

/* Stream prefetch with stream ID (0-7) */
#define DCBT_STREAM(addr, stream_id) \
    __asm__ __volatile__("dcbt %1, %0" : : "r"(addr), "i"(stream_id))

/* Prefetch entire weight tensor to L2/L3 resident */
static inline void dcbt_resident_weights(const void* base, size_t bytes) {
    const size_t CACHE_LINE = 128;  /* POWER8 cache line */
    const char* p = (const char*)base;
    const char* end = p + bytes;
    
    /* Prefetch every cache line as FULL RESIDENT */
    while (p < end) {
        DCBT_RESIDENT_FULL(p);
        p += CACHE_LINE;
    }
}

/* Prefetch Q/K/V for attention - all FULL RESIDENT */
static inline void dcbt_resident_qkv(
    const void* Q, const void* K, const void* V,
    size_t qkv_bytes
) {
    dcbt_resident_weights(Q, qkv_bytes);
    dcbt_resident_weights(K, qkv_bytes);
    dcbt_resident_weights(V, qkv_bytes);
}

/* Prefetch matmul weights - most critical path */
static inline void dcbt_resident_matmul(
    const void* weights,
    size_t weight_bytes,
    const void* input,
    size_t input_bytes
) {
    /* Weights are reused many times - FULL RESIDENT */
    dcbt_resident_weights(weights, weight_bytes);
    
    /* Input only used once - normal prefetch */
    const size_t CACHE_LINE = 128;
    const char* p = (const char*)input;
    const char* end = p + input_bytes;
    while (p < end) {
        DCBT_NORMAL(p);
        p += CACHE_LINE;
    }
}

#else  /* Not POWER8 */

#define DCBT_RESIDENT_FULL(addr) (void)(addr)
#define DCBT_RESIDENT_PARTIAL(addr) (void)(addr)
#define DCBT_NORMAL(addr) (void)(addr)
#define DCBT_STREAM(addr, id) (void)(addr)

static inline void dcbt_resident_weights(const void* base, size_t bytes) {
    (void)base; (void)bytes;
}
static inline void dcbt_resident_qkv(const void* Q, const void* K, const void* V, size_t qkv_bytes) {
    (void)Q; (void)K; (void)V; (void)qkv_bytes;
}
static inline void dcbt_resident_matmul(const void* weights, size_t weight_bytes,
                                        const void* input, size_t input_bytes) {
    (void)weights; (void)weight_bytes; (void)input; (void)input_bytes;
}

#endif  /* __powerpc64__ */

#endif  /* GGML_DCBT_RESIDENT_H */
