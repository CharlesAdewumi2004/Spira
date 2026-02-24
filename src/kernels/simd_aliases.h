#ifndef SPIRA_KERNELS_SIMD_ALIASES_H
#define SPIRA_KERNELS_SIMD_ALIASES_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace spira::kernel::simd {

// ============================================================================
// Register type aliases
// ============================================================================

// -- 128-bit registers (SSE) --
using reg128_float  = __m128;      // 4 x float
using reg128_double = __m128d;     // 2 x double
using reg128_int    = __m128i;     // 4 x int32, or 8 x int16, etc.

// -- 256-bit registers (AVX/AVX2) --
using reg256_float  = __m256;      // 8 x float
using reg256_double = __m256d;     // 4 x double
using reg256_int    = __m256i;     // 8 x int32, or 16 x int16, etc.

// -- 512-bit registers (AVX-512) --
using reg512_float  = __m512;      // 16 x float
using reg512_double = __m512d;     // 8 x double
using reg512_int    = __m512i;     // 16 x int32, etc.

// ============================================================================
// Zero initialisation
// ============================================================================

inline reg128_float  zero_float_128()  { return _mm_setzero_ps(); }
inline reg128_double zero_double_128() { return _mm_setzero_pd(); }
inline reg256_float  zero_float_256()  { return _mm256_setzero_ps(); }
inline reg256_double zero_double_256() { return _mm256_setzero_pd(); }

// ============================================================================
// Loads — contiguous memory into register
// ============================================================================

// -- 128-bit --
inline reg128_float  load_float_128(const float* p)   { return _mm_loadu_ps(p); }
inline reg128_double load_double_128(const double* p)  { return _mm_loadu_pd(p); }
inline reg128_int    load_int_128(const int* p)        { return _mm_loadu_si128((const __m128i*)p); }

// -- 256-bit --
inline reg256_float  load_float_256(const float* p)    { return _mm256_loadu_ps(p); }
inline reg256_double load_double_256(const double* p)  { return _mm256_loadu_pd(p); }
inline reg256_int    load_int_256(const int* p)        { return _mm256_loadu_si256((const __m256i*)p); }

// ============================================================================
// Stores — register back to memory
// ============================================================================

inline void store_float_128(float* p, reg128_float v)    { _mm_storeu_ps(p, v); }
inline void store_double_128(double* p, reg128_double v)  { _mm_storeu_pd(p, v); }
inline void store_float_256(float* p, reg256_float v)     { _mm256_storeu_ps(p, v); }
inline void store_double_256(double* p, reg256_double v)  { _mm256_storeu_pd(p, v); }

// ============================================================================
// Gather — scattered memory into register using indices
// ============================================================================

// AVX2: 4 doubles using 4 x int32 indices
inline reg256_double gather_double_256(const double* base, reg128_int indices) {
    return _mm256_i32gather_pd(base, indices, sizeof(double));
}

// AVX2: 8 floats using 8 x int32 indices
inline reg256_float gather_float_256(const float* base, reg256_int indices) {
    return _mm256_i32gather_ps(base, indices, sizeof(float));
}

// ============================================================================
// Arithmetic — element-wise (vertical)
// ============================================================================

// -- Add --
inline reg128_float  add_float_128(reg128_float a, reg128_float b)    { return _mm_add_ps(a, b); }
inline reg128_double add_double_128(reg128_double a, reg128_double b) { return _mm_add_pd(a, b); }
inline reg256_float  add_float_256(reg256_float a, reg256_float b)    { return _mm256_add_ps(a, b); }
inline reg256_double add_double_256(reg256_double a, reg256_double b) { return _mm256_add_pd(a, b); }

// -- Subtract --
inline reg128_float  sub_float_128(reg128_float a, reg128_float b)    { return _mm_sub_ps(a, b); }
inline reg128_double sub_double_128(reg128_double a, reg128_double b) { return _mm_sub_pd(a, b); }
inline reg256_float  sub_float_256(reg256_float a, reg256_float b)    { return _mm256_sub_ps(a, b); }
inline reg256_double sub_double_256(reg256_double a, reg256_double b) { return _mm256_sub_pd(a, b); }

// -- Multiply --
inline reg128_float  mul_float_128(reg128_float a, reg128_float b)    { return _mm_mul_ps(a, b); }
inline reg128_double mul_double_128(reg128_double a, reg128_double b) { return _mm_mul_pd(a, b); }
inline reg256_float  mul_float_256(reg256_float a, reg256_float b)    { return _mm256_mul_ps(a, b); }
inline reg256_double mul_double_256(reg256_double a, reg256_double b) { return _mm256_mul_pd(a, b); }

// -- Divide --
inline reg128_float  div_float_128(reg128_float a, reg128_float b)    { return _mm_div_ps(a, b); }
inline reg128_double div_double_128(reg128_double a, reg128_double b) { return _mm_div_pd(a, b); }
inline reg256_float  div_float_256(reg256_float a, reg256_float b)    { return _mm256_div_ps(a, b); }
inline reg256_double div_double_256(reg256_double a, reg256_double b) { return _mm256_div_pd(a, b); }

// ============================================================================
// Fused Multiply-Add — acc = a * b + acc (single rounding, more precise)
// ============================================================================

inline reg128_float  fma_float_128(reg128_float a, reg128_float b, reg128_float acc) {
    return _mm_fmadd_ps(a, b, acc);
}
inline reg128_double fma_double_128(reg128_double a, reg128_double b, reg128_double acc) {
    return _mm_fmadd_pd(a, b, acc);
}
inline reg256_float  fma_float_256(reg256_float a, reg256_float b, reg256_float acc) {
    return _mm256_fmadd_ps(a, b, acc);
}
inline reg256_double fma_double_256(reg256_double a, reg256_double b, reg256_double acc) {
    return _mm256_fmadd_pd(a, b, acc);
}

// ============================================================================
// Horizontal reduction — sum all lanes into a scalar
// ============================================================================

// Sum 4 floats in 128-bit register → single float
inline float reduce_sum_float_128(reg128_float v) {
    // [a, b, c, d]
    reg128_float hi = _mm_movehl_ps(v, v);            // [c, d, -, -]
    reg128_float sum1 = _mm_add_ps(v, hi);            // [a+c, b+d, -, -]
    reg128_float shuf = _mm_movehdup_ps(sum1);         // [b+d, b+d, -, -]
    reg128_float sum2 = _mm_add_ss(sum1, shuf);        // [a+c+b+d, -, -, -]
    return _mm_cvtss_f32(sum2);
}

// Sum 2 doubles in 128-bit register → single double
inline double reduce_sum_double_128(reg128_double v) {
    // [a, b]
    reg128_double hi = _mm_unpackhi_pd(v, v);          // [b, b]
    reg128_double sum = _mm_add_sd(v, hi);             // [a+b, -]
    return _mm_cvtsd_f64(sum);
}

// Sum 8 floats in 256-bit register → single float
inline float reduce_sum_float_256(reg256_float v) {
    // Split 256 → two 128
    reg128_float hi = _mm256_extractf128_ps(v, 1);     // upper 4 floats
    reg128_float lo = _mm256_castps256_ps128(v);        // lower 4 floats
    reg128_float sum = _mm_add_ps(lo, hi);              // 4 partial sums
    return reduce_sum_float_128(sum);                    // reduce to scalar
}

// Sum 4 doubles in 256-bit register → single double
inline double reduce_sum_double_256(reg256_double v) {
    // Split 256 → two 128
    reg128_double hi = _mm256_extractf128_pd(v, 1);    // upper 2 doubles
    reg128_double lo = _mm256_castpd256_pd128(v);       // lower 2 doubles
    reg128_double sum = _mm_add_pd(lo, hi);             // 2 partial sums
    return reduce_sum_double_128(sum);                   // reduce to scalar
}

// ============================================================================
// Broadcast — fill all lanes with one value
// ============================================================================

inline reg128_float  broadcast_float_128(float val)    { return _mm_set1_ps(val); }
inline reg128_double broadcast_double_128(double val)  { return _mm_set1_pd(val); }
inline reg256_float  broadcast_float_256(float val)    { return _mm256_set1_ps(val); }
inline reg256_double broadcast_double_256(double val)  { return _mm256_set1_pd(val); }

// ============================================================================
// Comparison — returns mask (all 1s or all 0s per lane)
// ============================================================================

inline reg256_float  cmp_eq_float_256(reg256_float a, reg256_float b)    { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
inline reg256_float  cmp_lt_float_256(reg256_float a, reg256_float b)    { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
inline reg256_float  cmp_gt_float_256(reg256_float a, reg256_float b)    { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
inline reg256_double cmp_eq_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
inline reg256_double cmp_lt_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
inline reg256_double cmp_gt_double_256(reg256_double a, reg256_double b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }

// ============================================================================
// Blend — branchless select: pick from a or b based on mask
// ============================================================================

inline reg256_float  blend_float_256(reg256_float a, reg256_float b, reg256_float mask) {
    return _mm256_blendv_ps(a, b, mask);   // mask lane all-1s → pick b, all-0s → pick a
}
inline reg256_double blend_double_256(reg256_double a, reg256_double b, reg256_double mask) {
    return _mm256_blendv_pd(a, b, mask);
}

// ============================================================================
// Min / Max — element-wise
// ============================================================================

inline reg256_float  min_float_256(reg256_float a, reg256_float b)    { return _mm256_min_ps(a, b); }
inline reg256_float  max_float_256(reg256_float a, reg256_float b)    { return _mm256_max_ps(a, b); }
inline reg256_double min_double_256(reg256_double a, reg256_double b) { return _mm256_min_pd(a, b); }
inline reg256_double max_double_256(reg256_double a, reg256_double b) { return _mm256_max_pd(a, b); }

} // namespace spira::kernel::simd

#endif // SPIRA_KERNELS_SIMD_ALIASES_H