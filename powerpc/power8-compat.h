/*
 * power8-compat.h - POWER8 vec_xl_len compatibility
 *
 * DO NOT define __POWER9_VECTOR__ - that enables GCC to use POWER9 builtins!
 * Instead, provide just the vec_xl_len shim for POWER8 use.
 */
#ifndef POWER8_COMPAT_H
#define POWER8_COMPAT_H

#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)

#include <altivec.h>
#include <string.h>

/*
 * vec_xl / vec_xst are native on POWER8 (lxvw4x/lxvd2x, stxvw4x/stxvd2x with
 * -mvsx) and handle unaligned addresses, so no shim is provided. Mapping
 * them to vec_ld/vec_st -- as an earlier version did whenever the compiler
 * did not expose vec_xl as a macro (e.g. clang) -- silently rounds the
 * address down to a 16-byte boundary and loads/stores the wrong bytes.
 */

/*
 * vec_xl_len(ptr, len) - partial vector load (POWER9 lxvl instruction).
 * On POWER8: copy min(len, 16) bytes into a zeroed 16-byte buffer and read
 * it back as the vector type vec_xl() would return for `ptr`.
 *
 * GCC's <altivec.h> (checked: GCC 15) #defines vec_xl_len to the POWER9-only
 * __builtin_vec_lxvl, so it must be #undef'd first: an #ifndef guard leaves
 * the POWER9 builtin in place (a hard error with -mcpu=power8), and a bare
 * redefinition emits a "redefined" warning (fatal under -Werror).
 * The union (rather than casting a char buffer to a vector pointer) keeps
 * the type pun well-defined under -O3 strict aliasing.
 */
#ifdef vec_xl_len
#undef vec_xl_len
#endif
#define vec_xl_len(ptr, len) \
    __extension__ ({ \
        union { \
            unsigned char __b[16]; \
            __typeof__(vec_xl(0, (ptr))) __v; \
        } __u; \
        size_t __len = (size_t)(len); \
        if (__len > 16) __len = 16; \
        __builtin_memset(__u.__b, 0, 16); \
        __builtin_memcpy(__u.__b, (ptr), __len); \
        __u.__v; \
    })

/* Marker that POWER8 compat is active */
#define GGML_POWER8_COMPAT_ACTIVE 1

#endif /* __POWER8_VECTOR__ && !__POWER9_VECTOR__ */

#endif /* POWER8_COMPAT_H */
