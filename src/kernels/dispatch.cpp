#include "hw_detect.hpp"
#include "kernels/dot_impls.hpp"
#include "spira/kernels/kernels.h"

namespace spira::kernel {
double (*sparse_dot_double)(const double *vals, const uint32_t *cols, const double *x, size_t n);
float (*sparse_dot_float)(const float *vals, const uint32_t *cols, const float *x, size_t n);
} // namespace spira::kernel

// Picks the best kernel for this CPU once, during static initialisation.
static struct KernelInit {
    KernelInit() {
        using namespace spira::kernel;
        const CpuFeatures cpu;

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
#elif defined(SPIRA_ARCH_ARM64)
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
