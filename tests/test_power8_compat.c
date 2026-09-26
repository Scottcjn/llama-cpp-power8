/*
 * Tests for power8-compat.h (built only for AltiVec/VSX targets).
 *
 * The same source is compiled twice by CMake, once against each copy of the
 * header (ggml/src/ggml-cpu/arch/powerpc/ and powerpc/), with -Werror so a
 * macro redefinition warning also fails the build.
 */
/* Include the header under test first: it must be self-contained. */
#include "power8-compat.h"

#include <altivec.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static unsigned char __attribute__((aligned(16))) src[64];

/* vec_xl must honour unaligned addresses (lxvw4x/lxvd2x on POWER8).
 * A vec_ld-based fallback silently rounds the address down to 16 bytes. */
static void test_vec_xl_unaligned(void)
{
    for (int off = 0; off < 16; off++) {
        vector unsigned char v = vec_xl(0, src + off);
        unsigned char out[16];
        memcpy(out, &v, sizeof out);
        CHECK(memcmp(out, src + off, 16) == 0);
    }
}

static void test_vec_xst_unaligned(void)
{
    unsigned char __attribute__((aligned(16))) dst[48];
    vector unsigned char v = vec_xl(0, src);
    for (int off = 0; off < 16; off++) {
        memset(dst, 0xAA, sizeof dst);
        vec_xst(v, 0, dst + off);
        CHECK(memcmp(dst + off, src, 16) == 0);
        for (int i = 0; i < off; i++) CHECK(dst[i] == 0xAA);
        for (int i = off + 16; i < (int)sizeof dst; i++) CHECK(dst[i] == 0xAA);
    }
}

/* vec_xl_len(ptr, n): first min(n,16) bytes from ptr, remainder zero. */
static void test_vec_xl_len_partial(void)
{
    const size_t lens[] = {0, 1, 4, 8, 12, 15, 16, 17, 255};
    for (size_t li = 0; li < sizeof lens / sizeof lens[0]; li++) {
        for (int off = 0; off < 16; off += 3) {
            size_t n = lens[li];
            size_t take = n > 16 ? 16 : n;
            vector unsigned char v = vec_xl_len(src + off, n);
            unsigned char out[16];
            memcpy(out, &v, sizeof out);
            CHECK(memcmp(out, src + off, take) == 0);
            for (size_t i = take; i < 16; i++) CHECK(out[i] == 0);
        }
    }
}

/* quants.c uses it on signed char scale arrays with small lengths. */
static void test_vec_xl_len_signed_type(void)
{
    const signed char s[12] = {-1, -2, -3, -4, 5, 6, 7, 8, 9, 10, 11, 12};
    vector signed char u0 = (vector signed char)vec_xl_len(s, 8);
    vector signed char u2 = (vector signed char)vec_xl_len(s + 8, 4);
    signed char a[16], b[16];
    memcpy(a, &u0, 16);
    memcpy(b, &u2, 16);
    CHECK(memcmp(a, s, 8) == 0);
    CHECK(memcmp(b, s + 8, 4) == 0);
    for (int i = 8; i < 16; i++) CHECK(a[i] == 0);
    for (int i = 4; i < 16; i++) CHECK(b[i] == 0);
}

int main(void)
{
    for (int i = 0; i < (int)sizeof src; i++) src[i] = (unsigned char)(i * 7 + 1);

    test_vec_xl_unaligned();
    test_vec_xst_unaligned();
    test_vec_xl_len_partial();
    test_vec_xl_len_signed_type();

    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
