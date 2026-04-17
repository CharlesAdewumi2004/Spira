#ifndef SPIRA_KERNELS_SIMD_ALIASES_SSE_HPP
#define SPIRA_KERNELS_SIMD_ALIASES_SSE_HPP

#include <cstdint>
#include <immintrin.h>

namespace spira::kernel::simd
{

// ============================================================================
// SSE2 (baseline for x86-64)
// ============================================================================

using reg128_float = __m128;
using reg128_double = __m128d;
using reg128_int = __m128i;

// Zero
inline reg128_float zero_float_128() { return _mm_setzero_ps(); }
inline reg128_double zero_double_128() { return _mm_setzero_pd(); }
inline reg128_int zero_int_128() { return _mm_setzero_si128(); }

// Broadcast
inline reg128_float broadcast_float_128(float val) { return _mm_set1_ps(val); }
inline reg128_double broadcast_double_128(double val) { return _mm_set1_pd(val); }
inline reg128_int broadcast_int32_128(int val) { return _mm_set1_epi32(val); }
inline reg128_int broadcast_int64_128(int64_t val) { return _mm_set1_epi64x(val); }

// Pack
inline reg128_float pack_float_128(float l0, float l1, float l2, float l3) { return _mm_setr_ps(l0, l1, l2, l3); }
inline reg128_double pack_double_128(double l0, double l1) { return _mm_setr_pd(l0, l1); }
inline reg128_int pack_int32_128(int l0, int l1, int l2, int l3) { return _mm_setr_epi32(l0, l1, l2, l3); }

// Loads
inline reg128_float load_float_128(const float *p) { return _mm_loadu_ps(p); }
inline reg128_double load_double_128(const double *p) { return _mm_loadu_pd(p); }
inline reg128_int load_int_128(const int *p) { return _mm_loadu_si128((const __m128i *)p); }
inline reg128_int load_int_128(const uint32_t *p) { return _mm_loadu_si128((const __m128i *)p); }

// Stores
inline void store_float_128(float *p, reg128_float v) { _mm_storeu_ps(p, v); }
inline void store_double_128(double *p, reg128_double v) { _mm_storeu_pd(p, v); }
inline void store_int_128(void *p, reg128_int v) { _mm_storeu_si128((__m128i *)p, v); }

// Add
inline reg128_float add_float_128(reg128_float a, reg128_float b) { return _mm_add_ps(a, b); }
inline reg128_double add_double_128(reg128_double a, reg128_double b) { return _mm_add_pd(a, b); }
inline reg128_int add_int32_128(reg128_int a, reg128_int b) { return _mm_add_epi32(a, b); }
inline reg128_int add_int64_128(reg128_int a, reg128_int b) { return _mm_add_epi64(a, b); }

// Sub
inline reg128_float sub_float_128(reg128_float a, reg128_float b) { return _mm_sub_ps(a, b); }
inline reg128_double sub_double_128(reg128_double a, reg128_double b) { return _mm_sub_pd(a, b); }
inline reg128_int sub_int32_128(reg128_int a, reg128_int b) { return _mm_sub_epi32(a, b); }

// Mul
inline reg128_float mul_float_128(reg128_float a, reg128_float b) { return _mm_mul_ps(a, b); }
inline reg128_double mul_double_128(reg128_double a, reg128_double b) { return _mm_mul_pd(a, b); }

// Div
inline reg128_float div_float_128(reg128_float a, reg128_float b) { return _mm_div_ps(a, b); }
inline reg128_double div_double_128(reg128_double a, reg128_double b) { return _mm_div_pd(a, b); }

// Sqrt
inline reg128_float sqrt_float_128(reg128_float a) { return _mm_sqrt_ps(a); }
inline reg128_double sqrt_double_128(reg128_double a) { return _mm_sqrt_pd(a); }

// Rcp / Rsqrt
inline reg128_float rcp_float_128(reg128_float a) { return _mm_rcp_ps(a); }
inline reg128_float rsqrt_float_128(reg128_float a) { return _mm_rsqrt_ps(a); }

// Abs (float/double via bitmask - SSE2)
inline reg128_float abs_float_128(reg128_float a) {
    reg128_int mask = _mm_set1_epi32(0x7FFFFFFF);
    return _mm_and_ps(a, _mm_castsi128_ps(mask));
}
inline reg128_double abs_double_128(reg128_double a) {
    reg128_int mask = _mm_set1_epi64x(0x7FFFFFFFFFFFFFFF);
    return _mm_and_pd(a, _mm_castsi128_pd(mask));
}

// Neg
inline reg128_float neg_float_128(reg128_float a) { return sub_float_128(zero_float_128(), a); }
inline reg128_double neg_double_128(reg128_double a) { return sub_double_128(zero_double_128(), a); }

// Min/Max (float/double - SSE/SSE2)
inline reg128_float min_float_128(reg128_float a, reg128_float b) { return _mm_min_ps(a, b); }
inline reg128_float max_float_128(reg128_float a, reg128_float b) { return _mm_max_ps(a, b); }
inline reg128_double min_double_128(reg128_double a, reg128_double b) { return _mm_min_pd(a, b); }
inline reg128_double max_double_128(reg128_double a, reg128_double b) { return _mm_max_pd(a, b); }

// Compare (float - SSE, double - SSE2)
inline reg128_float cmp_eq_float_128(reg128_float a, reg128_float b) { return _mm_cmpeq_ps(a, b); }
inline reg128_float cmp_lt_float_128(reg128_float a, reg128_float b) { return _mm_cmplt_ps(a, b); }
inline reg128_float cmp_gt_float_128(reg128_float a, reg128_float b) { return _mm_cmpgt_ps(a, b); }
inline reg128_float cmp_le_float_128(reg128_float a, reg128_float b) { return _mm_cmple_ps(a, b); }
inline reg128_float cmp_ge_float_128(reg128_float a, reg128_float b) { return _mm_cmpge_ps(a, b); }

inline reg128_double cmp_eq_double_128(reg128_double a, reg128_double b) { return _mm_cmpeq_pd(a, b); }
inline reg128_double cmp_lt_double_128(reg128_double a, reg128_double b) { return _mm_cmplt_pd(a, b); }
inline reg128_double cmp_gt_double_128(reg128_double a, reg128_double b) { return _mm_cmpgt_pd(a, b); }
inline reg128_double cmp_le_double_128(reg128_double a, reg128_double b) { return _mm_cmple_pd(a, b); }
inline reg128_double cmp_ge_double_128(reg128_double a, reg128_double b) { return _mm_cmpge_pd(a, b); }

inline reg128_int cmp_eq_int32_128(reg128_int a, reg128_int b) { return _mm_cmpeq_epi32(a, b); }
inline reg128_int cmp_gt_int32_128(reg128_int a, reg128_int b) { return _mm_cmpgt_epi32(a, b); }

// Movemask
inline int movemask_float_128(reg128_float a) { return _mm_movemask_ps(a); }
inline int movemask_double_128(reg128_double a) { return _mm_movemask_pd(a); }
inline int movemask_int8_128(reg128_int a) { return _mm_movemask_epi8(a); }

// Bitwise
inline reg128_float bitand_float_128(reg128_float a, reg128_float b) { return _mm_and_ps(a, b); }
inline reg128_double bitand_double_128(reg128_double a, reg128_double b) { return _mm_and_pd(a, b); }
inline reg128_int bitand_int_128(reg128_int a, reg128_int b) { return _mm_and_si128(a, b); }

inline reg128_float bitor_float_128(reg128_float a, reg128_float b) { return _mm_or_ps(a, b); }
inline reg128_double bitor_double_128(reg128_double a, reg128_double b) { return _mm_or_pd(a, b); }
inline reg128_int bitor_int_128(reg128_int a, reg128_int b) { return _mm_or_si128(a, b); }

inline reg128_float bitxor_float_128(reg128_float a, reg128_float b) { return _mm_xor_ps(a, b); }
inline reg128_double bitxor_double_128(reg128_double a, reg128_double b) { return _mm_xor_pd(a, b); }
inline reg128_int bitxor_int_128(reg128_int a, reg128_int b) { return _mm_xor_si128(a, b); }

inline reg128_float bitandnot_float_128(reg128_float a, reg128_float b) { return _mm_andnot_ps(a, b); }
inline reg128_double bitandnot_double_128(reg128_double a, reg128_double b) { return _mm_andnot_pd(a, b); }
inline reg128_int bitandnot_int_128(reg128_int a, reg128_int b) { return _mm_andnot_si128(a, b); }

// Shifts (SSE2)
inline reg128_int shl_int32_128(reg128_int a, int count) { return _mm_slli_epi32(a, count); }
inline reg128_int shl_int64_128(reg128_int a, int count) { return _mm_slli_epi64(a, count); }

inline reg128_int shr_int32_128(reg128_int a, int count) { return _mm_srai_epi32(a, count); }

inline reg128_int shrl_int32_128(reg128_int a, int count) { return _mm_srli_epi32(a, count); }
inline reg128_int shrl_int64_128(reg128_int a, int count) { return _mm_srli_epi64(a, count); }

// Casts
inline reg128_float cast_int_to_float_128(reg128_int a) { return _mm_castsi128_ps(a); }
inline reg128_double cast_int_to_double_128(reg128_int a) { return _mm_castsi128_pd(a); }
inline reg128_int cast_float_to_int_128(reg128_float a) { return _mm_castps_si128(a); }
inline reg128_int cast_double_to_int_128(reg128_double a) { return _mm_castpd_si128(a); }

// Convert (SSE2)
inline reg128_float cvt_int32_to_float_128(reg128_int a) { return _mm_cvtepi32_ps(a); }
inline reg128_int cvt_float_to_int32_128(reg128_float a) { return _mm_cvttps_epi32(a); }

inline reg128_double cvt_float_to_double_128(reg128_float a) { return _mm_cvtps_pd(a); }
inline reg128_float cvt_double_to_float_128(reg128_double a) { return _mm_cvtpd_ps(a); }

// Reductions (SSE2 - uses _mm_unpackhi_pd)
inline double reduce_sum_double_128(reg128_double v) {
    reg128_double hi = _mm_unpackhi_pd(v, v);
    reg128_double sum = _mm_add_sd(v, hi);
    return _mm_cvtsd_f64(sum);
}

inline double reduce_min_double_128(reg128_double v) {
    reg128_double hi = _mm_unpackhi_pd(v, v);
    reg128_double min = _mm_min_sd(v, hi);
    return _mm_cvtsd_f64(min);
}

inline double reduce_max_double_128(reg128_double v) {
    reg128_double hi = _mm_unpackhi_pd(v, v);
    reg128_double max = _mm_max_sd(v, hi);
    return _mm_cvtsd_f64(max);
}

// ============================================================================
// SSE3 (requires __SSE3__)
// ============================================================================
#if defined(__SSE3__)

// Float reductions use _mm_movehdup_ps (SSE3) and _mm_movehl_ps (SSE)
inline float reduce_sum_float_128(reg128_float v) {
    reg128_float hi = _mm_movehl_ps(v, v);
    reg128_float sum1 = _mm_add_ps(v, hi);
    reg128_float shuf = _mm_movehdup_ps(sum1);
    reg128_float sum2 = _mm_add_ss(sum1, shuf);
    return _mm_cvtss_f32(sum2);
}

inline float reduce_min_float_128(reg128_float v) {
    reg128_float hi = _mm_movehl_ps(v, v);
    reg128_float min1 = _mm_min_ps(v, hi);
    reg128_float shuf = _mm_movehdup_ps(min1);
    reg128_float min2 = _mm_min_ss(min1, shuf);
    return _mm_cvtss_f32(min2);
}

inline float reduce_max_float_128(reg128_float v) {
    reg128_float hi = _mm_movehl_ps(v, v);
    reg128_float max1 = _mm_max_ps(v, hi);
    reg128_float shuf = _mm_movehdup_ps(max1);
    reg128_float max2 = _mm_max_ss(max1, shuf);
    return _mm_cvtss_f32(max2);
}

#endif // __SSE3__

// ============================================================================
// SSSE3 (requires __SSSE3__)
// ============================================================================
#if defined(__SSSE3__)

inline reg128_int abs_int32_128(reg128_int a) { return _mm_abs_epi32(a); }

#endif // __SSSE3__

// ============================================================================
// SSE4.1 (requires __SSE4_1__)
// ============================================================================
#if defined(__SSE4_1__)

// Integer mul (SSE4.1)
inline reg128_int mullo_int32_128(reg128_int a, reg128_int b) { return _mm_mullo_epi32(a, b); }

// Integer min/max (SSE4.1)
inline reg128_int min_int32_128(reg128_int a, reg128_int b) { return _mm_min_epi32(a, b); }
inline reg128_int max_int32_128(reg128_int a, reg128_int b) { return _mm_max_epi32(a, b); }

// Blend (SSE4.1)
inline reg128_float blend_float_128(reg128_float a, reg128_float b, reg128_float mask) {
    return _mm_blendv_ps(a, b, mask);
}
inline reg128_double blend_double_128(reg128_double a, reg128_double b, reg128_double mask) {
    return _mm_blendv_pd(a, b, mask);
}

#endif // __SSE4_1__

// ============================================================================
// FMA (requires __FMA__)
// ============================================================================
#if defined(__FMA__)

inline reg128_float fma_float_128(reg128_float a, reg128_float b, reg128_float c) { return _mm_fmadd_ps(a, b, c); }
inline reg128_double fma_double_128(reg128_double a, reg128_double b, reg128_double c) { return _mm_fmadd_pd(a, b, c); }

inline reg128_float fms_float_128(reg128_float a, reg128_float b, reg128_float c) { return _mm_fmsub_ps(a, b, c); }
inline reg128_double fms_double_128(reg128_double a, reg128_double b, reg128_double c) { return _mm_fmsub_pd(a, b, c); }

inline reg128_float fnma_float_128(reg128_float a, reg128_float b, reg128_float c) { return _mm_fnmadd_ps(a, b, c); }
inline reg128_double fnma_double_128(reg128_double a, reg128_double b, reg128_double c) { return _mm_fnmadd_pd(a, b, c); }

#endif // __FMA__

} // namespace spira::kernel::simd

#endif