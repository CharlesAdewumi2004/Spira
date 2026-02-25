#ifndef SPIRA_KERNELS_SIMD_ALIASES_AVX_H
#define SPIRA_KERNELS_SIMD_ALIASES_AVX_H

#include <cstdint>
#include <immintrin.h>

#include "simd_sse_aliases.h"

namespace spira::kernel::simd {

// ============================================================================
// AVX (requires __AVX__)
// 256-bit float and double operations only.
// Integer 256-bit ops require AVX2 - see simd_avx2_aliases.h
// ============================================================================
#if defined(__AVX__)

using reg256_float = __m256;
using reg256_double = __m256d;
using reg256_int = __m256i;

// Zero
inline reg256_float zero_float_256() { return _mm256_setzero_ps(); }
inline reg256_double zero_double_256() { return _mm256_setzero_pd(); }
inline reg256_int zero_int_256() { return _mm256_setzero_si256(); }

// Broadcast
inline reg256_float broadcast_float_256(float val) { return _mm256_set1_ps(val); }
inline reg256_double broadcast_double_256(double val) { return _mm256_set1_pd(val); }

// Pack
inline reg256_float pack_float_256(float l0, float l1, float l2, float l3, float l4, float l5, float l6, float l7) {
    return _mm256_setr_ps(l0, l1, l2, l3, l4, l5, l6, l7);
}
inline reg256_double pack_double_256(double l0, double l1, double l2, double l3) {
    return _mm256_setr_pd(l0, l1, l2, l3);
}

// Loads
inline reg256_float load_float_256(const float *p) { return _mm256_loadu_ps(p); }
inline reg256_double load_double_256(const double *p) { return _mm256_loadu_pd(p); }

// Stores
inline void store_float_256(float *p, reg256_float v) { _mm256_storeu_ps(p, v); }
inline void store_double_256(double *p, reg256_double v) { _mm256_storeu_pd(p, v); }

// Add
inline reg256_float add_float_256(reg256_float a, reg256_float b) { return _mm256_add_ps(a, b); }
inline reg256_double add_double_256(reg256_double a, reg256_double b) { return _mm256_add_pd(a, b); }

// Sub
inline reg256_float sub_float_256(reg256_float a, reg256_float b) { return _mm256_sub_ps(a, b); }
inline reg256_double sub_double_256(reg256_double a, reg256_double b) { return _mm256_sub_pd(a, b); }

// Mul
inline reg256_float mul_float_256(reg256_float a, reg256_float b) { return _mm256_mul_ps(a, b); }
inline reg256_double mul_double_256(reg256_double a, reg256_double b) { return _mm256_mul_pd(a, b); }

// Div
inline reg256_float div_float_256(reg256_float a, reg256_float b) { return _mm256_div_ps(a, b); }
inline reg256_double div_double_256(reg256_double a, reg256_double b) { return _mm256_div_pd(a, b); }

// Sqrt
inline reg256_float sqrt_float_256(reg256_float a) { return _mm256_sqrt_ps(a); }
inline reg256_double sqrt_double_256(reg256_double a) { return _mm256_sqrt_pd(a); }

// Rcp / Rsqrt
inline reg256_float rcp_float_256(reg256_float a) { return _mm256_rcp_ps(a); }
inline reg256_float rsqrt_float_256(reg256_float a) { return _mm256_rsqrt_ps(a); }

// Abs (float/double via bitmask - AVX)
inline reg256_float abs_float_256(reg256_float a) {
    reg256_int mask = _mm256_set1_epi32(0x7FFFFFFF);
    return _mm256_and_ps(a, _mm256_castsi256_ps(mask));
}
inline reg256_double abs_double_256(reg256_double a) {
    reg256_int mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
    return _mm256_and_pd(a, _mm256_castsi256_pd(mask));
}

// Neg
inline reg256_float neg_float_256(reg256_float a) { return sub_float_256(zero_float_256(), a); }
inline reg256_double neg_double_256(reg256_double a) { return sub_double_256(zero_double_256(), a); }

// Min/Max
inline reg256_float min_float_256(reg256_float a, reg256_float b) { return _mm256_min_ps(a, b); }
inline reg256_float max_float_256(reg256_float a, reg256_float b) { return _mm256_max_ps(a, b); }
inline reg256_double min_double_256(reg256_double a, reg256_double b) { return _mm256_min_pd(a, b); }
inline reg256_double max_double_256(reg256_double a, reg256_double b) { return _mm256_max_pd(a, b); }

// Compare (AVX)
inline reg256_float cmp_eq_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
inline reg256_float cmp_lt_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
inline reg256_float cmp_gt_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
inline reg256_float cmp_le_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
inline reg256_float cmp_ge_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
inline reg256_float cmp_neq_float_256(reg256_float a, reg256_float b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ); }

inline reg256_double cmp_eq_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
inline reg256_double cmp_lt_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
inline reg256_double cmp_gt_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
inline reg256_double cmp_le_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
inline reg256_double cmp_ge_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); }
inline reg256_double cmp_neq_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ); }

// Blend (AVX)
inline reg256_float blend_float_256(reg256_float a, reg256_float b, reg256_float mask) {
    return _mm256_blendv_ps(a, b, mask);
}
inline reg256_double blend_double_256(reg256_double a, reg256_double b, reg256_double mask) {
    return _mm256_blendv_pd(a, b, mask);
}

// Movemask
inline int movemask_float_256(reg256_float a) { return _mm256_movemask_ps(a); }
inline int movemask_double_256(reg256_double a) { return _mm256_movemask_pd(a); }

// Bitwise (float/double - AVX)
inline reg256_float bitand_float_256(reg256_float a, reg256_float b) { return _mm256_and_ps(a, b); }
inline reg256_double bitand_double_256(reg256_double a, reg256_double b) { return _mm256_and_pd(a, b); }

inline reg256_float bitor_float_256(reg256_float a, reg256_float b) { return _mm256_or_ps(a, b); }
inline reg256_double bitor_double_256(reg256_double a, reg256_double b) { return _mm256_or_pd(a, b); }

inline reg256_float bitxor_float_256(reg256_float a, reg256_float b) { return _mm256_xor_ps(a, b); }
inline reg256_double bitxor_double_256(reg256_double a, reg256_double b) { return _mm256_xor_pd(a, b); }

inline reg256_float bitandnot_float_256(reg256_float a, reg256_float b) { return _mm256_andnot_ps(a, b); }
inline reg256_double bitandnot_double_256(reg256_double a, reg256_double b) { return _mm256_andnot_pd(a, b); }

// Casts (AVX - reinterpret, no codegen)
inline reg256_float cast_int_to_float_256(reg256_int a) { return _mm256_castsi256_ps(a); }
inline reg256_double cast_int_to_double_256(reg256_int a) { return _mm256_castsi256_pd(a); }
inline reg256_int cast_float_to_int_256(reg256_float a) { return _mm256_castps_si256(a); }
inline reg256_int cast_double_to_int_256(reg256_double a) { return _mm256_castpd_si256(a); }

inline reg128_float cast_256_to_128_float(reg256_float a) { return _mm256_castps256_ps128(a); }
inline reg128_double cast_256_to_128_double(reg256_double a) { return _mm256_castpd256_pd128(a); }
inline reg128_float extract_high_128_float(reg256_float a) { return _mm256_extractf128_ps(a, 1); }
inline reg128_double extract_high_128_double(reg256_double a) { return _mm256_extractf128_pd(a, 1); }

// Convert (AVX)
inline reg256_double cvt_float_to_double_256(reg128_float a) { return _mm256_cvtps_pd(a); }
inline reg128_float cvt_double_to_float_256(reg256_double a) { return _mm256_cvtpd_ps(a); }

// Reductions (AVX - float/double, reuse SSE3 reducers)
#if defined(__SSE3__)

inline float reduce_sum_float_256(reg256_float v) {
    reg128_float hi = _mm256_extractf128_ps(v, 1);
    reg128_float lo = _mm256_castps256_ps128(v);
    reg128_float sum = _mm_add_ps(lo, hi);
    return reduce_sum_float_128(sum);
}

inline float reduce_min_float_256(reg256_float v) {
    reg128_float hi = _mm256_extractf128_ps(v, 1);
    reg128_float lo = _mm256_castps256_ps128(v);
    reg128_float min = _mm_min_ps(lo, hi);
    return reduce_min_float_128(min);
}

inline float reduce_max_float_256(reg256_float v) {
    reg128_float hi = _mm256_extractf128_ps(v, 1);
    reg128_float lo = _mm256_castps256_ps128(v);
    reg128_float max = _mm_max_ps(lo, hi);
    return reduce_max_float_128(max);
}

#endif // __SSE3__

inline double reduce_sum_double_256(reg256_double v) {
    reg128_double hi = _mm256_extractf128_pd(v, 1);
    reg128_double lo = _mm256_castpd256_pd128(v);
    reg128_double sum = _mm_add_pd(lo, hi);
    return reduce_sum_double_128(sum);
}

inline double reduce_min_double_256(reg256_double v) {
    reg128_double hi = _mm256_extractf128_pd(v, 1);
    reg128_double lo = _mm256_castpd256_pd128(v);
    reg128_double min = _mm_min_pd(lo, hi);
    return reduce_min_double_128(min);
}

inline double reduce_max_double_256(reg256_double v) {
    reg128_double hi = _mm256_extractf128_pd(v, 1);
    reg128_double lo = _mm256_castpd256_pd128(v);
    reg128_double max = _mm_max_pd(lo, hi);
    return reduce_max_double_128(max);
}

// ============================================================================
// FMA + AVX (requires __FMA__)
// ============================================================================
#if defined(__FMA__)

inline reg256_float fma_float_256(reg256_float a, reg256_float b, reg256_float c) { return _mm256_fmadd_ps(a, b, c); }
inline reg256_double fma_double_256(reg256_double a, reg256_double b, reg256_double c) {
    return _mm256_fmadd_pd(a, b, c);
}

inline reg256_float fms_float_256(reg256_float a, reg256_float b, reg256_float c) { return _mm256_fmsub_ps(a, b, c); }
inline reg256_double fms_double_256(reg256_double a, reg256_double b, reg256_double c) {
    return _mm256_fmsub_pd(a, b, c);
}

inline reg256_float fnma_float_256(reg256_float a, reg256_float b, reg256_float c) { return _mm256_fnmadd_ps(a, b, c); }
inline reg256_double fnma_double_256(reg256_double a, reg256_double b, reg256_double c) {
    return _mm256_fnmadd_pd(a, b, c);
}

#endif // __FMA__

#endif // __AVX__

// ============================================================================
// AVX2 (requires __AVX2__)
// 256-bit integer operations, gathers, variable shifts
// ============================================================================
#if defined(__AVX2__)

// Broadcast (integer)
inline reg256_int broadcast_int32_256(int val) { return _mm256_set1_epi32(val); }
inline reg256_int broadcast_int64_256(int64_t val) { return _mm256_set1_epi64x(val); }

// Pack (integer)
inline reg256_int pack_int32_256(int l0, int l1, int l2, int l3, int l4, int l5, int l6, int l7) {
    return _mm256_setr_epi32(l0, l1, l2, l3, l4, l5, l6, l7);
}

// Loads (integer)
inline reg256_int load_int_256(const int *p) { return _mm256_loadu_si256((const __m256i *)p); }
inline reg256_int load_int_256(const uint32_t *p) { return _mm256_loadu_si256((const __m256i *)p); }

// Stores (integer)
inline void store_int_256(void *p, reg256_int v) { _mm256_storeu_si256((__m256i *)p, v); }

// Gather
inline reg256_double gather_double_256(const double *base, reg128_int indices) {
    return _mm256_i32gather_pd(base, indices, sizeof(double));
}

inline reg256_float gather_float_256(const float *base, reg256_int indices) {
    return _mm256_i32gather_ps(base, indices, sizeof(float));
}

inline reg256_int gather_int32_256(const int *base, reg256_int indices) {
    return _mm256_i32gather_epi32(base, indices, sizeof(int));
}

// Add (integer)
inline reg256_int add_int32_256(reg256_int a, reg256_int b) { return _mm256_add_epi32(a, b); }
inline reg256_int add_int64_256(reg256_int a, reg256_int b) { return _mm256_add_epi64(a, b); }

// Sub (integer)
inline reg256_int sub_int32_256(reg256_int a, reg256_int b) { return _mm256_sub_epi32(a, b); }

// Mul (integer)
inline reg256_int mullo_int32_256(reg256_int a, reg256_int b) { return _mm256_mullo_epi32(a, b); }

// Abs (integer)
inline reg256_int abs_int32_256(reg256_int a) { return _mm256_abs_epi32(a); }

// Min/Max (integer)
inline reg256_int min_int32_256(reg256_int a, reg256_int b) { return _mm256_min_epi32(a, b); }
inline reg256_int max_int32_256(reg256_int a, reg256_int b) { return _mm256_max_epi32(a, b); }

// Compare (integer)
inline reg256_int cmp_eq_int32_256(reg256_int a, reg256_int b) { return _mm256_cmpeq_epi32(a, b); }
inline reg256_int cmp_gt_int32_256(reg256_int a, reg256_int b) { return _mm256_cmpgt_epi32(a, b); }

// Movemask (integer)
inline int movemask_int8_256(reg256_int a) { return _mm256_movemask_epi8(a); }

// Bitwise (integer)
inline reg256_int bitand_int_256(reg256_int a, reg256_int b) { return _mm256_and_si256(a, b); }
inline reg256_int bitor_int_256(reg256_int a, reg256_int b) { return _mm256_or_si256(a, b); }
inline reg256_int bitxor_int_256(reg256_int a, reg256_int b) { return _mm256_xor_si256(a, b); }
inline reg256_int bitandnot_int_256(reg256_int a, reg256_int b) { return _mm256_andnot_si256(a, b); }

// Shifts (immediate)
inline reg256_int shl_int32_256(reg256_int a, int count) { return _mm256_slli_epi32(a, count); }
inline reg256_int shl_int64_256(reg256_int a, int count) { return _mm256_slli_epi64(a, count); }

inline reg256_int shr_int32_256(reg256_int a, int count) { return _mm256_srai_epi32(a, count); }

inline reg256_int shrl_int32_256(reg256_int a, int count) { return _mm256_srli_epi32(a, count); }
inline reg256_int shrl_int64_256(reg256_int a, int count) { return _mm256_srli_epi64(a, count); }

// Variable shifts (AVX2)
inline reg256_int shlv_int32_256(reg256_int a, reg256_int counts) { return _mm256_sllv_epi32(a, counts); }
inline reg256_int shrv_int32_256(reg256_int a, reg256_int counts) { return _mm256_srav_epi32(a, counts); }

// Convert (integer <-> float, AVX2)
inline reg256_float cvt_int32_to_float_256(reg256_int a) { return _mm256_cvtepi32_ps(a); }
inline reg256_int cvt_float_to_int32_256(reg256_float a) { return _mm256_cvttps_epi32(a); }

#endif // __AVX2__

} // namespace spira::kernel::simd

#endif