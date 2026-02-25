#include "cpu_detect.h"
#include "spira/kernels/kernels.h"
#include <stddef.h>

// ---- Always available (all platforms) ----
extern double sparse_dot_double_scalar(const double *, const uint32_t *, const double *, size_t);
extern float sparse_dot_float_scalar(const float *, const uint32_t *, const float *, size_t);

#if defined(SPIRA_ARCH_X86)
// ---- x86: SSE4.2 ----
extern double sparse_dot_double_sse(const double *, const uint32_t *, const double *, size_t);
extern float sparse_dot_float_sse(const float *, const uint32_t *, const float *, size_t);

// ---- x86: AVX2 + FMA ----
extern double sparse_dot_double_avx(const double *, const uint32_t *, const double *, size_t);
extern float sparse_dot_float_avx(const float *, const uint32_t *, const float *, size_t);

// ---- x86: AVX-512 ----
extern double sparse_dot_double_avx512(const double *, const uint32_t *, const double *, size_t);
extern float sparse_dot_float_avx512(const float *, const uint32_t *, const float *, size_t);
#endif

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
// ---- ARM: NEON ----
extern double sparse_dot_double_neon(const double *, const uint32_t *, const double *, size_t);
extern float sparse_dot_float_neon(const float *, const uint32_t *, const float *, size_t);
#endif

namespace spira::kernel {
double (*sparse_dot_double)(const double *vals, const uint32_t *cols, const double *x, size_t n);
float (*sparse_dot_float)(const float *vals, const uint32_t *cols, const float *x, size_t n);
} // namespace spira::kernel

static struct KernelInit {
    KernelInit() {
        using namespace spira::kernel;
        CpuFeatures cpu;

#if defined(SPIRA_ARCH_X86)
        if (cpu.avx512f) {
            sparse_dot_double = sparse_dot_double_avx512;
            sparse_dot_float = sparse_dot_float_avx512;
        } else if (cpu.avx2 && cpu.fma) {
            sparse_dot_double = sparse_dot_double_avx;
            sparse_dot_float = sparse_dot_float_avx;
        } else if (cpu.sse42) {
            sparse_dot_double = sparse_dot_double_sse;
            sparse_dot_float = sparse_dot_float_sse;
        } else {
            sparse_dot_double = sparse_dot_double_scalar;
            sparse_dot_float = sparse_dot_float_scalar;
        }

#elif defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
        if (cpu.neon) {
            sparse_dot_double = sparse_dot_double_neon;
            sparse_dot_float = sparse_dot_float_neon;
        } else {
            sparse_dot_double = sparse_dot_double_scalar;
            sparse_dot_float = sparse_dot_float_scalar;
        }
#else
        sparse_dot_double = sparse_dot_double_scalar;
        sparse_dot_float = sparse_dot_float_scalar;
#endif
    }
} kernel_init;
