#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

double sparse_dot_double_neon(const double *vals, const uint32_t *cols,
                              const double *x, size_t n, size_t /*x_size*/)
{
    size_t i = 0;
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    for (; i + 8 <= n; i += 8)
    {
        float64x2_t v0 = vld1q_f64(vals + i);
        double tmp0[2] = {x[cols[i]], x[cols[i + 1]]};
        float64x2_t xv0 = vld1q_f64(tmp0);
        acc0 = vfmaq_f64(acc0, v0, xv0);

        float64x2_t v1 = vld1q_f64(vals + i + 2);
        double tmp1[2] = {x[cols[i + 2]], x[cols[i + 3]]};
        float64x2_t xv1 = vld1q_f64(tmp1);
        acc1 = vfmaq_f64(acc1, v1, xv1);

        float64x2_t v2 = vld1q_f64(vals + i + 4);
        double tmp2[2] = {x[cols[i + 4]], x[cols[i + 5]]};
        float64x2_t xv2 = vld1q_f64(tmp2);
        acc2 = vfmaq_f64(acc2, v2, xv2);

        float64x2_t v3 = vld1q_f64(vals + i + 6);
        double tmp3[2] = {x[cols[i + 6]], x[cols[i + 7]]};
        float64x2_t xv3 = vld1q_f64(tmp3);
        acc3 = vfmaq_f64(acc3, v3, xv3);
    }

    for (; i + 2 <= n; i += 2)
    {
        float64x2_t v = vld1q_f64(vals + i);
        double tmp[2] = {x[cols[i]], x[cols[i + 1]]};
        float64x2_t xv = vld1q_f64(tmp);
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

float sparse_dot_float_neon(const float *vals, const uint32_t *cols,
                            const float *x, size_t n, size_t /*x_size*/)
{
    size_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (; i + 16 <= n; i += 16)
    {
        float32x4_t v0 = vld1q_f32(vals + i);
        float tmp0[4] = {x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]]};
        float32x4_t xv0 = vld1q_f32(tmp0);
        acc0 = vfmaq_f32(acc0, v0, xv0);

        float32x4_t v1 = vld1q_f32(vals + i + 4);
        float tmp1[4] = {x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]]};
        float32x4_t xv1 = vld1q_f32(tmp1);
        acc1 = vfmaq_f32(acc1, v1, xv1);

        float32x4_t v2 = vld1q_f32(vals + i + 8);
        float tmp2[4] = {x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]]};
        float32x4_t xv2 = vld1q_f32(tmp2);
        acc2 = vfmaq_f32(acc2, v2, xv2);

        float32x4_t v3 = vld1q_f32(vals + i + 12);
        float tmp3[4] = {x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]};
        float32x4_t xv3 = vld1q_f32(tmp3);
        acc3 = vfmaq_f32(acc3, v3, xv3);
    }

    for (; i + 4 <= n; i += 4)
    {
        float32x4_t v = vld1q_f32(vals + i);
        float tmp[4] = {x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]]};
        float32x4_t xv = vld1q_f32(tmp);
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
