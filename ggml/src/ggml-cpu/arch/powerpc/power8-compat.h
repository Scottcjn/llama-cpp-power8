/*
 * power8-compat.h - POWER8 compatibility shims for POWER9-only intrinsics
 * Elyan Labs 2025
 *
 * POWER8 has: vec_xl, vec_xst, vec_extract, vec_insert, vec_perm, vec_msum
 * POWER9 only: vec_xl_len (partial load)
 */
#ifndef POWER8_COMPAT_H
#define POWER8_COMPAT_H

#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)

#include <altivec.h>
#include <string.h>

/* Enable POWER9 code paths - we provide the missing vec_xl_len */
#define __POWER9_VECTOR__ 1
#define GGML_POWER8_COMPAT_ACTIVE 1

/*
 * vec_xl_len - partial vector load (POWER9 lxvl instruction)
 * On POWER8, use memcpy into aligned buffer then load.
 * Returns vector type matching the pointer type.
 */
#define vec_xl_len(ptr, len) \
    __extension__ ({ \
        __attribute__((aligned(16))) unsigned char __buf[16] = {0}; \
        size_t __len = (len) > 16 ? 16 : (size_t)(len); \
        memcpy(__buf, (ptr), __len); \
        *((__typeof__(vec_xl(0, (ptr)))*)__buf); \
    })

#endif /* __POWER8_VECTOR__ && !__POWER9_VECTOR__ */

#endif /* POWER8_COMPAT_H */
