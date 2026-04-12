#include <immintrin.h>
#include <cstddef>
#include <cstdint>

#if defined(SPIRA_ARCH_X86)

double sparse_dot_double_avx512(const double* vals, const uint32_t* cols,
                                const double* x, size_t n, size_t /*x_size*/)
{
    size_t i = 0;
    __m512d acc_reg = _mm512_setzero_pd();

    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(vals + i);
        __m256i idx = _mm256_loadu_si256((const __m256i *)(cols + i));
        __m512d xv = _mm512_i32gather_pd(idx, x, 8);
        acc_reg = _mm512_fmadd_pd(v, xv, acc_reg);
    }

    double acc = _mm512_reduce_add_pd(acc_reg);

    for (; i < n; i++)
        acc += x[cols[i]] * vals[i];

    return acc;
}

float sparse_dot_float_avx512(const float* vals, const uint32_t* cols,
                              const float* x, size_t n, size_t /*x_size*/)
{
    size_t i = 0;
    __m512 acc_reg = _mm512_setzero_ps();

    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(vals + i);
        __m512i idx = _mm512_loadu_si512((const __m512i *)(cols + i));
        __m512 xv = _mm512_i32gather_ps(idx, x, 4);
        acc_reg = _mm512_fmadd_ps(v, xv, acc_reg);
    }

    float acc = _mm512_reduce_add_ps(acc_reg);

    for (; i < n; i++)
        acc += x[cols[i]] * vals[i];

    return acc;
}

#endif
