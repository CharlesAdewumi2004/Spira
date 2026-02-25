#include "../src/kernels/cpu_detect.h"

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

double sparse_dot_double_neon(const double *vals, const uint32_t *cols, const double *x, size_t n) {
    size_t i = 0;
    float64x2_t acc = vdupq_n_f64(0.0); // [0.0, 0.0]

    // 2 doubles per 128-bit register
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(vals + i);

        // No gather in NEON — manual load
        double tmp[2] = {x[cols[i]], x[cols[i + 1]]};
        float64x2_t xv = vld1q_f64(tmp);

        // NEON has FMA
        acc = vfmaq_f64(acc, v, xv);
    }

    // Horizontal sum — 2 lanes
    double result = vaddvq_f64(acc);

    // Scalar tail — 0 or 1 remaining
    for (; i < n; i++) {
        result += vals[i] * x[cols[i]];
    }

    return result;
}

float sparse_dot_float_neon(const float *vals, const uint32_t *cols, const float *x, size_t n) {
    size_t i = 0;
    float32x4_t acc = vdupq_n_f32(0.0f); // [0, 0, 0, 0]

    // 4 floats per 128-bit register
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(vals + i);

        // No gather in NEON — manual load
        float tmp[4] = {x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]]};
        float32x4_t xv = vld1q_f32(tmp);

        // NEON has FMA
        acc = vfmaq_f32(acc, v, xv);
    }

    // Horizontal sum — 4 lanes
    float result = vaddvq_f32(acc);

    // Scalar tail — 0 to 3 remaining
    for (; i < n; i++) {
        result += vals[i] * x[cols[i]];
    }

    return result;
}

#endif