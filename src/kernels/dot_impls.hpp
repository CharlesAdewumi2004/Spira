#pragma once

// Every sparse dot kernel built into spira_kernels. Private to the kernel
// library and its tests; users call the dispatched pointers in kernels.h.
// The x86 and ARM variants exist only when their source files are compiled
// for the matching architecture (see CMakeLists.txt).

#include <cstddef>
#include <cstdint>

double sparse_dot_double_scalar(const double *, const uint32_t *, const double *, size_t);
float sparse_dot_float_scalar(const float *, const uint32_t *, const float *, size_t);

double sparse_dot_double_sse(const double *, const uint32_t *, const double *, size_t);
float sparse_dot_float_sse(const float *, const uint32_t *, const float *, size_t);

double sparse_dot_double_avx(const double *, const uint32_t *, const double *, size_t);
float sparse_dot_float_avx(const float *, const uint32_t *, const float *, size_t);

double sparse_dot_double_avx512(const double *, const uint32_t *, const double *, size_t);
float sparse_dot_float_avx512(const float *, const uint32_t *, const float *, size_t);

double sparse_dot_double_neon(const double *, const uint32_t *, const double *, size_t);
float sparse_dot_float_neon(const float *, const uint32_t *, const float *, size_t);
