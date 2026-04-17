#ifndef SPIRA_KERNELS_SIMD_ALIASES_HPP
#define SPIRA_KERNELS_SIMD_ALIASES_HPP

#include <cstdint>
#include <immintrin.h>

namespace spira::kernel::simd
{

// ============================================================================
// Register type aliases
// ============================================================================

// -- 128-bit registers (SSE) --
using reg128_float = __m128;
using reg128_double = __m128d;
using reg128_int = __m128i;

// -- 256-bit registers (AVX/AVX2) --
using reg256_float = __m256;
using reg256_double = __m256d;
using reg256_int = __m256i;

// -- 512-bit registers (AVX-512) --
#ifdef __AVX512F__
using reg512_float = __m512;
using reg512_double = __m512d;
using reg512_int = __m512i;
#endif

// ============================================================================
// Zero initialisation
// ============================================================================

// -- 128-bit --
inline reg128_float zero_float_128() { return _mm_setzero_ps(); }
inline reg128_double zero_double_128() { return _mm_setzero_pd(); }
inline reg128_int zero_int_128() { return _mm_setzero_si128(); }

// -- 256-bit --
inline reg256_float zero_float_256() { return _mm256_setzero_ps(); }
inline reg256_double zero_double_256() { return _mm256_setzero_pd(); }
inline reg256_int zero_int_256() { return _mm256_setzero_si256(); }

// -- 512-bit --
#ifdef __AVX512F__
inline reg512_float zero_float_512() { return _mm512_setzero_ps(); }
inline reg512_double zero_double_512() { return _mm512_setzero_pd(); }
inline reg512_int zero_int_512() { return _mm512_setzero_si512(); }
#endif

// ============================================================================
// Broadcast — fill all lanes with one value
// ============================================================================

// -- 128-bit --
inline reg128_float broadcast_float_128(float val) { return _mm_set1_ps(val); }
inline reg128_double broadcast_double_128(double val) { return _mm_set1_pd(val); }
inline reg128_int broadcast_int32_128(int val) { return _mm_set1_epi32(val); }
inline reg128_int broadcast_int64_128(int64_t val) { return _mm_set1_epi64x(val); }

// -- 256-bit --
inline reg256_float broadcast_float_256(float val) { return _mm256_set1_ps(val); }
inline reg256_double broadcast_double_256(double val) { return _mm256_set1_pd(val); }
inline reg256_int broadcast_int32_256(int val) { return _mm256_set1_epi32(val); }
inline reg256_int broadcast_int64_256(int64_t val) { return _mm256_set1_epi64x(val); }

// -- 512-bit --
#ifdef __AVX512F__
inline reg512_float broadcast_float_512(float val) { return _mm512_set1_ps(val); }
inline reg512_double broadcast_double_512(double val) { return _mm512_set1_pd(val); }
inline reg512_int broadcast_int32_512(int val) { return _mm512_set1_epi32(val); }
inline reg512_int broadcast_int64_512(int64_t val) { return _mm512_set1_epi64(val); }
#endif

// ============================================================================
// Pack from scalars — construct register from individual values
// Arguments in NATURAL order (lane0 first)
// ============================================================================

// -- 128-bit --
inline reg128_float pack_float_128(float l0, float l1, float l2, float l3) { return _mm_setr_ps(l0, l1, l2, l3); }
inline reg128_double pack_double_128(double l0, double l1) { return _mm_setr_pd(l0, l1); }
inline reg128_int pack_int32_128(int l0, int l1, int l2, int l3) { return _mm_setr_epi32(l0, l1, l2, l3); }

// -- 256-bit --
inline reg256_float pack_float_256(float l0, float l1, float l2, float l3, float l4, float l5, float l6, float l7) {
    return _mm256_setr_ps(l0, l1, l2, l3, l4, l5, l6, l7);
}
inline reg256_double pack_double_256(double l0, double l1, double l2, double l3) {
    return _mm256_setr_pd(l0, l1, l2, l3);
}
inline reg256_int pack_int32_256(int l0, int l1, int l2, int l3, int l4, int l5, int l6, int l7) {
    return _mm256_setr_epi32(l0, l1, l2, l3, l4, l5, l6, l7);
}

// ============================================================================
// Loads — contiguous memory into register (unaligned)
// ============================================================================

// -- 128-bit --
inline reg128_float load_float_128(const float *p) { return _mm_loadu_ps(p); }
inline reg128_double load_double_128(const double *p) { return _mm_loadu_pd(p); }
inline reg128_int load_int_128(const int *p) { return _mm_loadu_si128((const __m128i *)p); }
inline reg128_int load_int_128(const uint32_t *p) { return _mm_loadu_si128((const __m128i *)p); }

// -- 256-bit --
inline reg256_float load_float_256(const float *p) { return _mm256_loadu_ps(p); }
inline reg256_double load_double_256(const double *p) { return _mm256_loadu_pd(p); }
inline reg256_int load_int_256(const int *p) { return _mm256_loadu_si256((const __m256i *)p); }
inline reg256_int load_int_256(const uint32_t *p) { return _mm256_loadu_si256((const __m256i *)p); }

// -- 512-bit --
#ifdef __AVX512F__
inline reg512_float load_float_512(const float *p) { return _mm512_loadu_ps(p); }
inline reg512_double load_double_512(const double *p) { return _mm512_loadu_pd(p); }
inline reg512_int load_int_512(const int *p) { return _mm512_loadu_si512((const __m512i *)p); }
inline reg512_int load_int_512(const uint32_t *p) { return _mm512_loadu_si512((const __m512i *)p); }
#endif

// ============================================================================
// Stores — register back to memory (unaligned)
// ============================================================================

// -- 128-bit --
inline void store_float_128(float *p, reg128_float v) { _mm_storeu_ps(p, v); }
inline void store_double_128(double *p, reg128_double v) { _mm_storeu_pd(p, v); }
inline void store_int_128(void *p, reg128_int v) { _mm_storeu_si128((__m128i *)p, v); }

// -- 256-bit --
inline void store_float_256(float *p, reg256_float v) { _mm256_storeu_ps(p, v); }
inline void store_double_256(double *p, reg256_double v) { _mm256_storeu_pd(p, v); }
inline void store_int_256(void *p, reg256_int v) { _mm256_storeu_si256((__m256i *)p, v); }

// -- 512-bit --
#ifdef __AVX512F__
inline void store_float_512(float *p, reg512_float v) { _mm512_storeu_ps(p, v); }
inline void store_double_512(double *p, reg512_double v) { _mm512_storeu_pd(p, v); }
inline void store_int_512(void *p, reg512_int v) { _mm512_storeu_si512((__m512i *)p, v); }
#endif

// ============================================================================
// Gather — scattered memory into register using int32 indices
// ============================================================================

// AVX2: 4 doubles from 4 x int32 indices (128-bit index reg)
inline reg256_double gather_double_256(const double *base, reg128_int indices) {
    return _mm256_i32gather_pd(base, indices, sizeof(double));
}

// AVX2: 8 floats from 8 x int32 indices (256-bit index reg)
inline reg256_float gather_float_256(const float *base, reg256_int indices) {
    return _mm256_i32gather_ps(base, indices, sizeof(float));
}

// AVX2: 4 int32s from 4 x int32 indices (128-bit index reg)
inline reg128_int gather_int32_128(const int *base, reg128_int indices) {
    return _mm_i32gather_epi32(base, indices, sizeof(int));
}

// AVX2: 8 int32s from 8 x int32 indices (256-bit index reg)
inline reg256_int gather_int32_256(const int *base, reg256_int indices) {
    return _mm256_i32gather_epi32(base, indices, sizeof(int));
}

#ifdef __AVX512F__
// AVX-512: 8 doubles from 8 x int32 indices (256-bit index reg)
inline reg512_double gather_double_512(const double *base, reg256_int indices) {
    return _mm512_i32gather_pd(indices, base, sizeof(double));
}

// AVX-512: 16 floats from 16 x int32 indices (512-bit index reg)
inline reg512_float gather_float_512(const float *base, reg512_int indices) {
    return _mm512_i32gather_ps(indices, base, sizeof(float));
}
#endif

// ============================================================================
// Scatter — register to scattered memory using int32 indices (AVX-512 only)
// ============================================================================

#ifdef __AVX512F__
inline void scatter_double_512(double *base, reg256_int indices, reg512_double vals) {
    _mm512_i32scatter_pd(base, indices, vals, sizeof(double));
}

inline void scatter_float_512(float *base, reg512_int indices, reg512_float vals) {
    _mm512_i32scatter_ps(base, indices, vals, sizeof(float));
}
#endif

// ============================================================================
// Add — element-wise
// ============================================================================

// -- float --
inline reg128_float add_float_128(reg128_float a, reg128_float b) { return _mm_add_ps(a, b); }
inline reg256_float add_float_256(reg256_float a, reg256_float b) { return _mm256_add_ps(a, b); }
#ifdef __AVX512F__
inline reg512_float add_float_512(reg512_float a, reg512_float b) { return _mm512_add_ps(a, b); }
#endif

// -- double --
inline reg128_double add_double_128(reg128_double a, reg128_double b) { return _mm_add_pd(a, b); }
inline reg256_double add_double_256(reg256_double a, reg256_double b) { return _mm256_add_pd(a, b); }
#ifdef __AVX512F__
inline reg512_double add_double_512(reg512_double a, reg512_double b) { return _mm512_add_pd(a, b); }
#endif

// -- int32 --
inline reg128_int add_int32_128(reg128_int a, reg128_int b) { return _mm_add_epi32(a, b); }
inline reg256_int add_int32_256(reg256_int a, reg256_int b) { return _mm256_add_epi32(a, b); }
#ifdef __AVX512F__
inline reg512_int add_int32_512(reg512_int a, reg512_int b) { return _mm512_add_epi32(a, b); }
#endif

// -- int64 --
inline reg128_int add_int64_128(reg128_int a, reg128_int b) { return _mm_add_epi64(a, b); }
inline reg256_int add_int64_256(reg256_int a, reg256_int b) { return _mm256_add_epi64(a, b); }
#ifdef __AVX512F__
inline reg512_int add_int64_512(reg512_int a, reg512_int b) { return _mm512_add_epi64(a, b); }
#endif

// ============================================================================
// Subtract — element-wise
// ============================================================================

// -- float --
inline reg128_float sub_float_128(reg128_float a, reg128_float b) { return _mm_sub_ps(a, b); }
inline reg256_float sub_float_256(reg256_float a, reg256_float b) { return _mm256_sub_ps(a, b); }
#ifdef __AVX512F__
inline reg512_float sub_float_512(reg512_float a, reg512_float b) { return _mm512_sub_ps(a, b); }
#endif

// -- double --
inline reg128_double sub_double_128(reg128_double a, reg128_double b) { return _mm_sub_pd(a, b); }
inline reg256_double sub_double_256(reg256_double a, reg256_double b) { return _mm256_sub_pd(a, b); }
#ifdef __AVX512F__
inline reg512_double sub_double_512(reg512_double a, reg512_double b) { return _mm512_sub_pd(a, b); }
#endif

// -- int32 --
inline reg128_int sub_int32_128(reg128_int a, reg128_int b) { return _mm_sub_epi32(a, b); }
inline reg256_int sub_int32_256(reg256_int a, reg256_int b) { return _mm256_sub_epi32(a, b); }
#ifdef __AVX512F__
inline reg512_int sub_int32_512(reg512_int a, reg512_int b) { return _mm512_sub_epi32(a, b); }
#endif

// ============================================================================
// Multiply — element-wiseMasked FMA(AVX - 512 only) r
// ============================================================================

// -- float --
inline reg128_float mul_float_128(reg128_float a, reg128_float b) { return _mm_mul_ps(a, b); }
inline reg256_float mul_float_256(reg256_float a, reg256_float b) { return _mm256_mul_ps(a, b); }
#ifdef __AVX512F__
inline reg512_float mul_float_512(reg512_float a, reg512_float b) { return _mm512_mul_ps(a, b); }
#endif

// -- double --
inline reg128_double mul_double_128(reg128_double a, reg128_double b) { return _mm_mul_pd(a, b); }
inline reg256_double mul_double_256(reg256_double a, reg256_double b) { return _mm256_mul_pd(a, b); }
#ifdef __AVX512F__
inline reg512_double mul_double_512(reg512_double a, reg512_double b) { return _mm512_mul_pd(a, b); }
#endif

// -- int32 (low 32 bits of each 64-bit product) --
inline reg128_int mullo_int32_128(reg128_int a, reg128_int b) { return _mm_mullo_epi32(a, b); }
inline reg256_int mullo_int32_256(reg256_int a, reg256_int b) { return _mm256_mullo_epi32(a, b); }
#ifdef __AVX512F__
inline reg512_int mullo_int32_512(reg512_int a, reg512_int b) { return _mm512_mullo_epi32(a, b); }
#endif

// ============================================================================
// Divide — element-wise (float/double only, no integer divide in SIMD)
// ============================================================================

// -- float --
inline reg128_float div_float_128(reg128_float a, reg128_float b) { return _mm_div_ps(a, b); }
inline reg256_float div_float_256(reg256_float a, reg256_float b) { return _mm256_div_ps(a, b); }
#ifdef __AVX512F__
inline reg512_float div_float_512(reg512_float a, reg512_float b) { return _mm512_div_ps(a, b); }
#endif

// -- double --
inline reg128_double div_double_128(reg128_double a, reg128_double b) { return _mm_div_pd(a, b); }
inline reg256_double div_double_256(reg256_double a, reg256_double b) { return _mm256_div_pd(a, b); }
#ifdef __AVX512F__
inline reg512_double div_double_512(reg512_double a, reg512_double b) { return _mm512_div_pd(a, b); }
#endif

// ============================================================================
// Fused Multiply-Add — a * b + c (single rounding, more precise)
// Requires FMA (Haswell+), NOT available in pure SSE
// ============================================================================

// -- float --
inline reg128_float fma_float_128(reg128_float a, reg128_float b, reg128_float c) { return _mm_fmadd_ps(a, b, c); }
inline reg256_float fma_float_256(reg256_float a, reg256_float b, reg256_float c) { return _mm256_fmadd_ps(a, b, c); }
#ifdef __AVX512F__
inline reg512_float fma_float_512(reg512_float a, reg512_float b, reg512_float c) { return _mm512_fmadd_ps(a, b, c); }
#endif

// -- double --
inline reg128_double fma_double_128(reg128_double a, reg128_double b, reg128_double c) { return _mm_fmadd_pd(a, b, c); }
inline reg256_double fma_double_256(reg256_double a, reg256_double b, reg256_double c) {
    return _mm256_fmadd_pd(a, b, c);
}
#ifdef __AVX512F__
inline reg512_double fma_double_512(reg512_double a, reg512_double b, reg512_double c) {
    return _mm512_fmadd_pd(a, b, c);
}
#endif

// ============================================================================
// Fused Multiply-Subtract — a * b - c
// ============================================================================

// -- float --
inline reg128_float fms_float_128(reg128_float a, reg128_float b, reg128_float c) { return _mm_fmsub_ps(a, b, c); }
inline reg256_float fms_float_256(reg256_float a, reg256_float b, reg256_float c) { return _mm256_fmsub_ps(a, b, c); }
#ifdef __AVX512F__
inline reg512_float fms_float_512(reg512_float a, reg512_float b, reg512_float c) { return _mm512_fmsub_ps(a, b, c); }
#endif

// -- double --
inline reg128_double fms_double_128(reg128_double a, reg128_double b, reg128_double c) { return _mm_fmsub_pd(a, b, c); }
inline reg256_double fms_double_256(reg256_double a, reg256_double b, reg256_double c) {
    return _mm256_fmsub_pd(a, b, c);
}
#ifdef __AVX512F__
inline reg512_double fms_double_512(reg512_double a, reg512_double b, reg512_double c) {
    return _mm512_fmsub_pd(a, b, c);
}
#endif

// ============================================================================
// Fused Negate-Multiply-Add — -(a * b) + c
// ============================================================================

// -- float --
inline reg128_float fnma_float_128(reg128_float a, reg128_float b, reg128_float c) { return _mm_fnmadd_ps(a, b, c); }
inline reg256_float fnma_float_256(reg256_float a, reg256_float b, reg256_float c) { return _mm256_fnmadd_ps(a, b, c); }
#ifdef __AVX512F__
inline reg512_float fnma_float_512(reg512_float a, reg512_float b, reg512_float c) { return _mm512_fnmadd_ps(a, b, c); }
#endif

// -- double --
inline reg128_double fnma_double_128(reg128_double a, reg128_double b, reg128_double c) {
    return _mm_fnmadd_pd(a, b, c);
}
inline reg256_double fnma_double_256(reg256_double a, reg256_double b, reg256_double c) {
    return _mm256_fnmadd_pd(a, b, c);
}
#ifdef __AVX512F__
inline reg512_double fnma_double_512(reg512_double a, reg512_double b, reg512_double c) {
    return _mm512_fnmadd_pd(a, b, c);
}
#endif

// ==========================================Masked FMA(AVX - 512 only) r==================================
// Square Root — element-wise
// ============================================================================

// -- float --
inline reg128_float sqrt_float_128(reg128_float a) { return _mm_sqrt_ps(a); }
inline reg256_float sqrt_float_256(reg256_float a) { return _mm256_sqrt_ps(a); }
#ifdef __AVX512F__
inline reg512_float sqrt_float_512(reg512_float a) { return _mm512_sqrt_ps(a); }
#endif

// -- double --
inline reg128_double sqrt_double_128(reg128_double a) { return _mm_sqrt_pd(a); }
inline reg256_double sqrt_double_256(reg256_double a) { return _mm256_sqrt_pd(a); }
#ifdef __AVX512F__
inline reg512_double sqrt_double_512(reg512_double a) { return _mm512_sqrt_pd(a); }
#endif

// ============================================================================
// Reciprocal approximation — 1/x (float only, ~12-bit precision)
// ============================================================================

inline reg128_float rcp_float_128(reg128_float a) { return _mm_rcp_ps(a); }
inline reg256_float rcp_float_256(reg256_float a) { return _mm256_rcp_ps(a); }

// ============================================================================
// Reciprocal square root approximation — 1/sqrt(x) (float only, ~12-bit)
// ============================================================================

inline reg128_float rsqrt_float_128(reg128_float a) { return _mm_rsqrt_ps(a); }
inline reg256_float rsqrt_float_256(reg256_float a) { return _mm256_rsqrt_ps(a); }

// ============================================================================
// Absolute value
// ============================================================================

// Float: clear sign bit using bitwise AND with 0x7FFFFFFF
inline reg128_float abs_float_128(reg128_float a) {
    reg128_int mask = _mm_set1_epi32(0x7FFFFFFF);
    return _mm_and_ps(a, _mm_castsi128_ps(mask));
}
inline reg256_float abs_float_256(reg256_float a) {
    reg256_int mask = _mm256_set1_epi32(0x7FFFFFFF);
    return _mm256_and_ps(a, _mm256_castsi256_ps(mask));
}

// Double: clear sign bit using bitwise AND with 0x7FFFFFFFFFFFFFFF
inline reg128_double abs_double_128(reg128_double a) {
    reg128_int mask = _mm_set1_epi64x(0x7FFFFFFFFFFFFFFF);
    return _mm_and_pd(a, _mm_castsi128_pd(mask));
}
inline reg256_double abs_double_256(reg256_double a) {
    reg256_int mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
    return _mm256_and_pd(a, _mm256_castsi256_pd(mask));
}

// Int32: SSE/AVX2 have _mm_abs_epi32
inline reg128_int abs_int32_128(reg128_int a) { return _mm_abs_epi32(a); }
inline reg256_int abs_int32_256(reg256_int a) { return _mm256_abs_epi32(a); }

#ifdef __AVX512F__
inline reg512_float abs_float_512(reg512_float a) { return _mm512_abs_ps(a); }
inline reg512_double abs_double_512(reg512_double a) { return _mm512_abs_pd(a); }
inline reg512_int abs_int32_512(reg512_int a) { return _mm512_abs_epi32(a); }
#endif

// ============================================================================
// Negate — flip sign
// ============================================================================

inline reg128_float neg_float_128(reg128_float a) { return sub_float_128(zero_float_128(), a); }
inline reg128_double neg_double_128(reg128_double a) { return sub_double_128(zero_double_128(), a); }
inline reg256_float neg_float_256(reg256_float a) { return sub_float_256(zero_float_256(), a); }
inline reg256_double neg_double_256(reg256_double a) { return sub_double_256(zero_double_256(), a); }
#ifdef __AVX512F__
inline reg512_float neg_float_512(reg512_float a) { return sub_float_512(zero_float_512(), a); }
inline reg512_double neg_double_512(reg512_double a) { return sub_double_512(zero_double_512(), a); }
#endif

// ============================================================================
// Min / Max — element-wise
// ============================================================================

// -- float --
inline reg128_float min_float_128(reg128_float a, reg128_float b) { return _mm_min_ps(a, b); }
inline reg128_float max_float_128(reg128_float a, reg128_float b) { return _mm_max_ps(a, b); }
inline reg256_float min_float_256(reg256_float a, reg256_float b) { return _mm256_min_ps(a, b); }
inline reg256_float max_float_256(reg256_float a, reg256_float b) { return _mm256_max_ps(a, b); }
#ifdef __AVX512F__
inline reg512_float min_float_512(reg512_float a, reg512_float b) { return _mm512_min_ps(a, b); }
inline reg512_float max_float_512(reg512_float a, reg512_float b) { return _mm512_max_ps(a, b); }
#endif

// -- double --
inline reg128_double min_double_128(reg128_double a, reg128_double b) { return _mm_min_pd(a, b); }
inline reg128_double max_double_128(reg128_double a, reg128_double b) { return _mm_max_pd(a, b); }
inline reg256_double min_double_256(reg256_double a, reg256_double b) { return _mm256_min_pd(a, b); }
inline reg256_double max_double_256(reg256_double a, reg256_double b) { return _mm256_max_pd(a, b); }
#ifdef __AVX512F__
inline reg512_double min_double_512(reg512_double a, reg512_double b) { return _mm512_min_pd(a, b); }
inline reg512_double max_double_512(reg512_double a, reg512_double b) { return _mm512_max_pd(a, b); }
#endif

// -- int32 --
inline reg128_int min_int32_128(reg128_int a, reg128_int b) { return _mm_min_epi32(a, b); }
inline reg128_int max_int32_128(reg128_int a, reg128_int b) { return _mm_max_epi32(a, b); }
inline reg256_int min_int32_256(reg256_int a, reg256_int b) { return _mm256_min_epi32(a, b); }
inline reg256_int max_int32_256(reg256_int a, reg256_int b) { return _mm256_max_epi32(a, b); }
#ifdef __AVX512F__
inline reg512_int min_int32_512(reg512_int a, reg512_int b) { return _mm512_min_epi32(a, b); }
inline reg512_int max_int32_512(reg512_int a, reg512_int b) { return _mm512_max_epi32(a, b); }
#endif

// ============================================================================
// Comparison — returns mask
// SSE/AVX: mask is same-type register (all-1s or all-0s per lane)
// AVX-512: mask is __mmask type (1 bit per lane)
// ============================================================================

// -- float 128-bit --
inline reg128_float cmp_eq_float_128(reg128_float a, reg128_float b) { return _mm_cmpeq_ps(a, b); }
inline reg128_float cmp_lt_float_128(reg128_float a, reg128_float b) { return _mm_cmplt_ps(a, b); }
inline reg128_float cmp_gt_float_128(reg128_float a, reg128_float b) { return _mm_cmpgt_ps(a, b); }
inline reg128_float cmp_le_float_128(reg128_float a, reg128_float b) { return _mm_cmple_ps(a, b); }
inline reg128_float cmp_ge_float_128(reg128_float a, reg128_float b) { return _mm_cmpge_ps(a, b); }

// -- float 256-bit --
inline reg256_float cmp_eq_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
inline reg256_float cmp_lt_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
inline reg256_float cmp_gt_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
inline reg256_float cmp_le_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
inline reg256_float cmp_ge_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
inline reg256_float cmp_neq_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ); }

// -- double 128-bit --
inline reg128_double cmp_eq_double_128(reg128_double a, reg128_double b) { return _mm_cmpeq_pd(a, b); }
inline reg128_double cmp_lt_double_128(reg128_double a, reg128_double b) { return _mm_cmplt_pd(a, b); }
inline reg128_double cmp_gt_double_128(reg128_double a, reg128_double b) { return _mm_cmpgt_pd(a, b); }
inline reg128_double cmp_le_double_128(reg128_double a, reg128_double b) { return _mm_cmple_pd(a, b); }
inline reg128_double cmp_ge_double_128(reg128_double a, reg128_double b) { return _mm_cmpge_pd(a, b); }

// -- double 256-bit --
inline reg256_double cmp_eq_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
inline reg256_double cmp_lt_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
inline reg256_double cmp_gt_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
inline reg256_double cmp_le_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
inline reg256_double cmp_ge_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); }
inline reg256_double cmp_neq_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ); }

// -- int32 --
inline reg128_int cmp_eq_int32_128(reg128_int a, reg128_int b) { return _mm_cmpeq_epi32(a, b); }
inline reg128_int cmp_gt_int32_128(reg128_int a, reg128_int b) { return _mm_cmpgt_epi32(a, b); }
inline reg256_int cmp_eq_int32_256(reg256_int a, reg256_int b) { return _mm256_cmpeq_epi32(a, b); }
inline reg256_int cmp_gt_int32_256(reg256_int a, reg256_int b) { return _mm256_cmpgt_epi32(a, b); }

// AVX-512 comparisons return __mmask types
#ifdef __AVX512F__
inline __mmask16 cmp_eq_float_512(reg512_float a, reg512_float b) { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ); }
inline __mmask16 cmp_lt_float_512(reg512_float a, reg512_float b) { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ); }
inline __mmask16 cmp_gt_float_512(reg512_float a, reg512_float b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
inline __mmask8 cmp_eq_double_512(reg512_double a, reg512_double b) { return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ); }
inline __mmask8 cmp_lt_double_512(reg512_double a, reg512_double b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ); }
inline __mmask8 cmp_gt_double_512(reg512_double a, reg512_double b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); }
inline __mmask16 cmp_eq_int32_512(reg512_int a, reg512_int b) { return _mm512_cmpeq_epi32_mask(a, b); }
inline __mmask16 cmp_gt_int32_512(reg512_int a, reg512_int b) { return _mm512_cmpgt_epi32_mask(a, b); }
#endif

// ============================================================================
// Blend — branchless select: mask all-0s → a, mask all-1s → b
// ============================================================================

// -- 128-bit --
inline reg128_float blend_float_128(reg128_float a, reg128_float b, reg128_float mask) {
    return _mm_blendv_ps(a, b, mask);
}
inline reg128_double blend_double_128(reg128_double a, reg128_double b, reg128_double mask) {
    return _mm_blendv_pd(a, b, mask);
}

// -- 256-bit --
inline reg256_float blend_float_256(reg256_float a, reg256_float b, reg256_float mask) {
    return _mm256_blendv_ps(a, b, mask);
}
inline reg256_double blend_double_256(reg256_double a, reg256_double b, reg256_double mask) {
    return _mm256_blendv_pd(a, b, mask);
}

// -- 512-bit (uses __mmask) --
#ifdef __AVX512F__
inline reg512_float blend_float_512(reg512_float a, reg512_float b, __mmask16 mask) {
    return _mm512_mask_blend_ps(mask, a, b);
}
inline reg512_double blend_double_512(reg512_double a, reg512_double b, __mmask8 mask) {
    return _mm512_mask_blend_pd(mask, a, b);
}
#endif

// ============================================================================
// Movemask — extract comparison results to scalar int (1 bit per lane)
// ============================================================================

inline int movemask_float_128(reg128_float a) { return _mm_movemask_ps(a); }
inline int movemask_double_128(reg128_double a) { return _mm_movemask_pd(a); }
inline int movemask_float_256(reg256_float a) { return _mm256_movemask_ps(a); }
inline int movemask_double_256(reg256_double a) { return _mm256_movemask_pd(a); }
inline int movemask_int8_128(reg128_int a) { return _mm_movemask_epi8(a); }
inline int movemask_int8_256(reg256_int a) { return _mm256_movemask_epi8(a); }

// ============================================================================
// Bitwise — operate on raw bits regardless of lane interpretation
// ============================================================================

// -- AND --
inline reg128_float bitand_float_128(reg128_float a, reg128_float b) { return _mm_and_ps(a, b); }
inline reg128_double bitand_double_128(reg128_double a, reg128_double b) { return _mm_and_pd(a, b); }
inline reg128_int bitand_int_128(reg128_int a, reg128_int b) { return _mm_and_si128(a, b); }
inline reg256_float bitand_float_256(reg256_float a, reg256_float b) { return _mm256_and_ps(a, b); }
inline reg256_double bitand_double_256(reg256_double a, reg256_double b) { return _mm256_and_pd(a, b); }
inline reg256_int bitand_int_256(reg256_int a, reg256_int b) { return _mm256_and_si256(a, b); }
#ifdef __AVX512F__
inline reg512_int bitand_int_512(reg512_int a, reg512_int b) { return _mm512_and_si512(a, b); }
#endif

// -- OR --
inline reg128_float bitor_float_128(reg128_float a, reg128_float b) { return _mm_or_ps(a, b); }
inline reg128_double bitor_double_128(reg128_double a, reg128_double b) { return _mm_or_pd(a, b); }
inline reg128_int bitor_int_128(reg128_int a, reg128_int b) { return _mm_or_si128(a, b); }
inline reg256_float bitor_float_256(reg256_float a, reg256_float b) { return _mm256_or_ps(a, b); }
inline reg256_double bitor_double_256(reg256_double a, reg256_double b) { return _mm256_or_pd(a, b); }
inline reg256_int bitor_int_256(reg256_int a, reg256_int b) { return _mm256_or_si256(a, b); }
#ifdef __AVX512F__
inline reg512_int bitor_int_512(reg512_int a, reg512_int b) { return _mm512_or_si512(a, b); }
#endif

// -- XOR --
inline reg128_float bitxor_float_128(reg128_float a, reg128_float b) { return _mm_xor_ps(a, b); }
inline reg128_double bitxor_double_128(reg128_double a, reg128_double b) { return _mm_xor_pd(a, b); }
inline reg128_int bitxor_int_128(reg128_int a, reg128_int b) { return _mm_xor_si128(a, b); }
inline reg256_float bitxor_float_256(reg256_float a, reg256_float b) { return _mm256_xor_ps(a, b); }
inline reg256_double bitxor_double_256(reg256_double a, reg256_double b) { return _mm256_xor_pd(a, b); }
inline reg256_int bitxor_int_256(reg256_int a, reg256_int b) { return _mm256_xor_si256(a, b); }
#ifdef __AVX512F__
inline reg512_int bitxor_int_512(reg512_int a, reg512_int b) { return _mm512_xor_si512(a, b); }
#endif

// -- AND NOT — (~a) & b --
inline reg128_float bitandnot_float_128(reg128_float a, reg128_float b) { return _mm_andnot_ps(a, b); }
inline reg128_double bitandnot_double_128(reg128_double a, reg128_double b) { return _mm_andnot_pd(a, b); }
inline reg128_int bitandnot_int_128(reg128_int a, reg128_int b) { return _mm_andnot_si128(a, b); }
inline reg256_float bitandnot_float_256(reg256_float a, reg256_float b) { return _mm256_andnot_ps(a, b); }
inline reg256_double bitandnot_double_256(reg256_double a, reg256_double b) { return _mm256_andnot_pd(a, b); }
inline reg256_int bitandnot_int_256(reg256_int a, reg256_int b) { return _mm256_andnot_si256(a, b); }
#ifdef __AVX512F__
inline reg512_int bitandnot_int_512(reg512_int a, reg512_int b) { return _mm512_andnot_si512(a, b); }
#endif

// ============================================================================
// Shift — integer lanes
// ============================================================================

// -- Left shift by immediate --
inline reg128_int shl_int32_128(reg128_int a, int count) { return _mm_slli_epi32(a, count); }
inline reg128_int shl_int64_128(reg128_int a, int count) { return _mm_slli_epi64(a, count); }
inline reg256_int shl_int32_256(reg256_int a, int count) { return _mm256_slli_epi32(a, count); }
inline reg256_int shl_int64_256(reg256_int a, int count) { return _mm256_slli_epi64(a, count); }

// -- Right shift by immediate (arithmetic = sign-extending) --
inline reg128_int shr_int32_128(reg128_int a, int count) { return _mm_srai_epi32(a, count); }
inline reg256_int shr_int32_256(reg256_int a, int count) { return _mm256_srai_epi32(a, count); }

// -- Right shift by immediate (logical = zero-extending) --
inline reg128_int shrl_int32_128(reg128_int a, int count) { return _mm_srli_epi32(a, count); }
inline reg128_int shrl_int64_128(reg128_int a, int count) { return _mm_srli_epi64(a, count); }
inline reg256_int shrl_int32_256(reg256_int a, int count) { return _mm256_srli_epi32(a, count); }
inline reg256_int shrl_int64_256(reg256_int a, int count) { return _mm256_srli_epi64(a, count); }

// -- Variable shift (each lane shifted by different amount, AVX2) --
inline reg256_int shlv_int32_256(reg256_int a, reg256_int counts) { return _mm256_sllv_epi32(a, counts); }
inline reg256_int shrv_int32_256(reg256_int a, reg256_int counts) { return _mm256_srav_epi32(a, counts); }

// ============================================================================
// Type cast / reinterpret — zero cost, just changes how compiler sees the bits
// ============================================================================

// -- 128-bit --
inline reg128_float cast_int_to_float_128(reg128_int a) { return _mm_castsi128_ps(a); }
inline reg128_double cast_int_to_double_128(reg128_int a) { return _mm_castsi128_pd(a); }
inline reg128_int cast_float_to_int_128(reg128_float a) { return _mm_castps_si128(a); }
inline reg128_int cast_double_to_int_128(reg128_double a) { return _mm_castpd_si128(a); }

// -- 256-bit --
inline reg256_float cast_int_to_float_256(reg256_int a) { return _mm256_castsi256_ps(a); }
inline reg256_double cast_int_to_double_256(reg256_int a) { return _mm256_castsi256_pd(a); }
inline reg256_int cast_float_to_int_256(reg256_float a) { return _mm256_castps_si256(a); }
inline reg256_int cast_double_to_int_256(reg256_double a) { return _mm256_castpd_si256(a); }

// -- Width cast (256 ↔ 128) --
inline reg128_float cast_256_to_128_float(reg256_float a) { return _mm256_castps256_ps128(a); }
inline reg128_double cast_256_to_128_double(reg256_double a) { return _mm256_castpd256_pd128(a); }
inline reg128_float extract_high_128_float(reg256_float a) { return _mm256_extractf128_ps(a, 1); }
inline reg128_double extract_high_128_double(reg256_double a) { return _mm256_extractf128_pd(a, 1); }

// ============================================================================
// Convert — actually changes the data (not just reinterpret)
// ============================================================================

// int32 → float
inline reg128_float cvt_int32_to_float_128(reg128_int a) { return _mm_cvtepi32_ps(a); }
inline reg256_float cvt_int32_to_float_256(reg256_int a) { return _mm256_cvtepi32_ps(a); }

// float → int32 (truncate toward zero)
inline reg128_int cvt_float_to_int32_128(reg128_float a) { return _mm_cvttps_epi32(a); }
inline reg256_int cvt_float_to_int32_256(reg256_float a) { return _mm256_cvttps_epi32(a); }

// float ↔ double
inline reg128_double cvt_float_to_double_128(reg128_float a) { return _mm_cvtps_pd(a); }
inline reg128_float cvt_double_to_float_128(reg128_double a) { return _mm_cvtpd_ps(a); }
inline reg256_double cvt_float_to_double_256(reg128_float a) { return _mm256_cvtps_pd(a); }
inline reg128_float cvt_double_to_float_256(reg256_double a) { return _mm256_cvtpd_ps(a); }

// ============================================================================
// Horizontal reduction — sum all lanes into a scalar
// ============================================================================

// Sum 4 floats in 128-bit register
inline float reduce_sum_float_128(reg128_float v) {
    reg128_float hi = _mm_movehl_ps(v, v);
    reg128_float sum1 = _mm_add_ps(v, hi);
    reg128_float shuf = _mm_movehdup_ps(sum1);
    reg128_float sum2 = _mm_add_ss(sum1, shuf);
    return _mm_cvtss_f32(sum2);
}

// Sum 2 doubles in 128-bit register
inline double reduce_sum_double_128(reg128_double v) {
    reg128_double hi = _mm_unpackhi_pd(v, v);
    reg128_double sum = _mm_add_sd(v, hi);
    return _mm_cvtsd_f64(sum);
}

// Sum 8 floats in 256-bit register
inline float reduce_sum_float_256(reg256_float v) {
    reg128_float hi = _mm256_extractf128_ps(v, 1);
    reg128_float lo = _mm256_castps256_ps128(v);
    reg128_float sum = _mm_add_ps(lo, hi);
    return reduce_sum_float_128(sum);
}

// Sum 4 doubles in 256-bit register
inline double reduce_sum_double_256(reg256_double v) {
    reg128_double hi = _mm256_extractf128_pd(v, 1);
    reg128_double lo = _mm256_castpd256_pd128(v);
    reg128_double sum = _mm_add_pd(lo, hi);
    return reduce_sum_double_128(sum);
}

#ifdef __AVX512F__
// Sum 16 floats in 512-bit register (compiler pseudo-intrinsic)
inline float reduce_sum_float_512(reg512_float v) { return _mm512_reduce_add_ps(v); }

// Sum 8 doubles in 512-bit register
inline double reduce_sum_double_512(reg512_double v) { return _mm512_reduce_add_pd(v); }
#endif

// -- Min reduction --
inline float reduce_min_float_128(reg128_float v) {
    reg128_float hi = _mm_movehl_ps(v, v);
    reg128_float min1 = _mm_min_ps(v, hi);
    reg128_float shuf = _mm_movehdup_ps(min1);
    reg128_float min2 = _mm_min_ss(min1, shuf);
    return _mm_cvtss_f32(min2);
}

inline double reduce_min_double_128(reg128_double v) {
    reg128_double hi = _mm_unpackhi_pd(v, v);
    reg128_double min = _mm_min_sd(v, hi);
    return _mm_cvtsd_f64(min);
}

inline float reduce_min_float_256(reg256_float v) {
    reg128_float hi = _mm256_extractf128_ps(v, 1);
    reg128_float lo = _mm256_castps256_ps128(v);
    reg128_float min = _mm_min_ps(lo, hi);
    return reduce_min_float_128(min);
}

inline double reduce_min_double_256(reg256_double v) {
    reg128_double hi = _mm256_extractf128_pd(v, 1);
    reg128_double lo = _mm256_castpd256_pd128(v);
    reg128_double min = _mm_min_pd(lo, hi);
    return reduce_min_double_128(min);
}

#ifdef __AVX512F__
inline float reduce_min_float_512(reg512_float v) { return _mm512_reduce_min_ps(v); }
inline double reduce_min_double_512(reg512_double v) { return _mm512_reduce_min_pd(v); }
#endif

// -- Max reduction --
inline float reduce_max_float_128(reg128_float v) {
    reg128_float hi = _mm_movehl_ps(v, v);
    reg128_float max1 = _mm_max_ps(v, hi);
    reg128_float shuf = _mm_movehdup_ps(max1);
    reg128_float max2 = _mm_max_ss(max1, shuf);
    return _mm_cvtss_f32(max2);
}

inline double reduce_max_double_128(reg128_double v) {
    reg128_double hi = _mm_unpackhi_pd(v, v);
    reg128_double max = _mm_max_sd(v, hi);
    return _mm_cvtsd_f64(max);
}

inline float reduce_max_float_256(reg256_float v) {
    reg128_float hi = _mm256_extractf128_ps(v, 1);
    reg128_float lo = _mm256_castps256_ps128(v);
    reg128_float max = _mm_max_ps(lo, hi);
    return reduce_max_float_128(max);
}

inline double reduce_max_double_256(reg256_double v) {
    reg128_double hi = _mm256_extractf128_pd(v, 1);
    reg128_double lo = _mm256_castpd256_pd128(v);
    reg128_double max = _mm_max_pd(lo, hi);
    return reduce_max_double_128(max);
}

#ifdef __AVX512F__
inline float reduce_max_float_512(reg512_float v) { return _mm512_reduce_max_ps(v); }
inline double reduce_max_double_512(reg512_double v) { return _mm512_reduce_max_pd(v); }
#endif

// ============================================================================
// Masked loads/stores (AVX-512) — only process lanes where mask bit is set
// ============================================================================

#ifdef __AVX512F__
// Load with mask — unset lanes become zero
inline reg512_float maskload_float_512(__mmask16 k, const float *p) { return _mm512_maskz_loadu_ps(k, p); }
inline reg512_double maskload_double_512(__mmask8 k, const double *p) { return _mm512_maskz_loadu_pd(k, p); }

// Store with mask — only write lanes where mask bit is set
inline void maskstore_float_512(float *p, __mmask16 k, reg512_float v) { _mm512_mask_storeu_ps(p, k, v); }
inline void maskstore_double_512(double *p, __mmask8 k, reg512_double v) { _mm512_mask_storeu_pd(p, k, v); }

// Create mask for tail handling: mask = (1 << remaining) - 1
inline __mmask16 make_mask16(int count) { return (__mmask16)((1U << count) - 1); }
inline __mmask8 make_mask8(int count) { return (__mmask8)((1U << count) - 1); }
#endif

// ============================================================================
// AVX-512 masked FMA — only accumulate where mask is set
// ============================================================================

#ifdef __AVX512F__
inline reg512_float mask_fma_float_512(__mmask16 k, reg512_float a, reg512_float b, reg512_float c) {
    return _mm512_mask_fmadd_ps(a, k, b, c);
}
inline reg512_double mask_fma_double_512(__mmask8 k, reg512_double a, reg512_double b, reg512_double c) {
    return _mm512_mask_fmadd_pd(a, k, b, c);
}
#endif


} // namespace spira::kernel::simd

#endif // SPIRA_KERNELS_SIMD_ALIASES_HPP