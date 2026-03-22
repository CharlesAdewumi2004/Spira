#include "../src/kernels/runtime_config.hpp"

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

double sparse_dot_double_neon(const double *vals, const uint32_t *cols, const double *x, size_t n, size_t x_size) {
    size_t i = 0;
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    // 4-accumulator loop: 8 doubles per iteration (4 × 2-wide)
    for (; i + 8 <= n; i += 8) {
        float64x2_t v0 = vld1q_f64(vals + i);
        float64x2_t xv0 = vdupq_n_f64(0.0);
        xv0 = vsetq_lane_f64(x[cols[i]],     xv0, 0);
        xv0 = vsetq_lane_f64(x[cols[i + 1]], xv0, 1);
        acc0 = vfmaq_f64(acc0, v0, xv0);

        float64x2_t v1 = vld1q_f64(vals + i + 2);
        float64x2_t xv1 = vdupq_n_f64(0.0);
        xv1 = vsetq_lane_f64(x[cols[i + 2]], xv1, 0);
        xv1 = vsetq_lane_f64(x[cols[i + 3]], xv1, 1);
        acc1 = vfmaq_f64(acc1, v1, xv1);

        float64x2_t v2 = vld1q_f64(vals + i + 4);
        float64x2_t xv2 = vdupq_n_f64(0.0);
        xv2 = vsetq_lane_f64(x[cols[i + 4]], xv2, 0);
        xv2 = vsetq_lane_f64(x[cols[i + 5]], xv2, 1);
        acc2 = vfmaq_f64(acc2, v2, xv2);

        float64x2_t v3 = vld1q_f64(vals + i + 6);
        float64x2_t xv3 = vdupq_n_f64(0.0);
        xv3 = vsetq_lane_f64(x[cols[i + 6]], xv3, 0);
        xv3 = vsetq_lane_f64(x[cols[i + 7]], xv3, 1);
        acc3 = vfmaq_f64(acc3, v3, xv3);
    }

    // single-register cleanup
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(vals + i);
        float64x2_t xv = vdupq_n_f64(0.0);
        xv = vsetq_lane_f64(x[cols[i]],     xv, 0);
        xv = vsetq_lane_f64(x[cols[i + 1]], xv, 1);
        acc0 = vfmaq_f64(acc0, v, xv);
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);
    double result = vaddvq_f64(acc0);

    for (; i < n; i++)
        result += vals[i] * x[cols[i]];

    return result;
}

float sparse_dot_float_neon(const float *vals, const uint32_t *cols, const float *x, size_t n, size_t x_size) {
    size_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    // 4-accumulator loop: 16 floats per iteration (4 × 4-wide)
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(vals + i);
        float32x4_t xv0 = vdupq_n_f32(0.0f);
        xv0 = vsetq_lane_f32(x[cols[i]],     xv0, 0);
        xv0 = vsetq_lane_f32(x[cols[i + 1]], xv0, 1);
        xv0 = vsetq_lane_f32(x[cols[i + 2]], xv0, 2);
        xv0 = vsetq_lane_f32(x[cols[i + 3]], xv0, 3);
        acc0 = vfmaq_f32(acc0, v0, xv0);

        float32x4_t v1 = vld1q_f32(vals + i + 4);
        float32x4_t xv1 = vdupq_n_f32(0.0f);
        xv1 = vsetq_lane_f32(x[cols[i + 4]], xv1, 0);
        xv1 = vsetq_lane_f32(x[cols[i + 5]], xv1, 1);
        xv1 = vsetq_lane_f32(x[cols[i + 6]], xv1, 2);
        xv1 = vsetq_lane_f32(x[cols[i + 7]], xv1, 3);
        acc1 = vfmaq_f32(acc1, v1, xv1);

        float32x4_t v2 = vld1q_f32(vals + i + 8);
        float32x4_t xv2 = vdupq_n_f32(0.0f);
        xv2 = vsetq_lane_f32(x[cols[i + 8]],  xv2, 0);
        xv2 = vsetq_lane_f32(x[cols[i + 9]],  xv2, 1);
        xv2 = vsetq_lane_f32(x[cols[i + 10]], xv2, 2);
        xv2 = vsetq_lane_f32(x[cols[i + 11]], xv2, 3);
        acc2 = vfmaq_f32(acc2, v2, xv2);

        float32x4_t v3 = vld1q_f32(vals + i + 12);
        float32x4_t xv3 = vdupq_n_f32(0.0f);
        xv3 = vsetq_lane_f32(x[cols[i + 12]], xv3, 0);
        xv3 = vsetq_lane_f32(x[cols[i + 13]], xv3, 1);
        xv3 = vsetq_lane_f32(x[cols[i + 14]], xv3, 2);
        xv3 = vsetq_lane_f32(x[cols[i + 15]], xv3, 3);
        acc3 = vfmaq_f32(acc3, v3, xv3);
    }

    // single-register cleanup
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(vals + i);
        float32x4_t xv = vdupq_n_f32(0.0f);
        xv = vsetq_lane_f32(x[cols[i]],     xv, 0);
        xv = vsetq_lane_f32(x[cols[i + 1]], xv, 1);
        xv = vsetq_lane_f32(x[cols[i + 2]], xv, 2);
        xv = vsetq_lane_f32(x[cols[i + 3]], xv, 3);
        acc0 = vfmaq_f32(acc0, v, xv);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    float result = vaddvq_f32(acc0);

    for (; i < n; i++)
        result += vals[i] * x[cols[i]];

    return result;
}

#endif
