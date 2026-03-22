#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "../src/kernels/hw_detect.hpp"
#include "spira/kernels/kernels.h"

// ============================================================================
// Extern declarations — free functions (no namespace)
// ============================================================================

// Scalar — always compiled on every platform
extern double sparse_dot_double_scalar(const double *, const uint32_t *, const double *, size_t, size_t);
extern float sparse_dot_float_scalar(const float *, const uint32_t *, const float *, size_t, size_t);

#if defined(SPIRA_ARCH_X86)
extern double sparse_dot_double_sse(const double *, const uint32_t *, const double *, size_t, size_t);
extern float sparse_dot_float_sse(const float *, const uint32_t *, const float *, size_t, size_t);

extern double sparse_dot_double_avx(const double *, const uint32_t *, const double *, size_t, size_t);
extern float sparse_dot_float_avx(const float *, const uint32_t *, const float *, size_t, size_t);

extern double sparse_dot_double_avx512(const double *, const uint32_t *, const double *, size_t, size_t);
extern float sparse_dot_float_avx512(const float *, const uint32_t *, const float *, size_t, size_t);
#endif

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
extern double sparse_dot_double_neon(const double *, const uint32_t *, const double *, size_t, size_t);
extern float sparse_dot_float_neon(const float *, const uint32_t *, const float *, size_t, size_t);
#endif

// ============================================================================
// Runtime CPU feature check — cached singleton
// ============================================================================

static const spira::kernel::CpuFeatures &get_cpu() {
    static spira::kernel::CpuFeatures cpu;
    return cpu;
}

// ============================================================================
// Reference implementation — plain C++, no SIMD, no FMA
// ============================================================================

static double reference_dot_double(const double *vals, const uint32_t *cols, const double *x, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        acc += vals[i] * x[cols[i]];
    }
    return acc;
}

static float reference_dot_float(const float *vals, const uint32_t *cols, const float *x, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        acc += vals[i] * x[cols[i]];
    }
    return acc;
}

// ============================================================================
// Test data generators
// ============================================================================

struct DoubleTestData {
    std::vector<double> vals;
    std::vector<uint32_t> cols;
    std::vector<double> x;
    size_t nnz;
};

struct FloatTestData {
    std::vector<float> vals;
    std::vector<uint32_t> cols;
    std::vector<float> x;
    size_t nnz;
};

static DoubleTestData make_double_data(size_t nnz, size_t x_size, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);
    std::uniform_int_distribution<uint32_t> col_dist(0, static_cast<uint32_t>(x_size - 1));

    DoubleTestData d;
    d.nnz = nnz;
    d.vals.resize(nnz);
    d.cols.resize(nnz);
    d.x.resize(x_size);

    for (size_t i = 0; i < nnz; i++) {
        d.vals[i] = val_dist(gen);
        d.cols[i] = col_dist(gen);
    }
    for (size_t i = 0; i < x_size; i++) {
        d.x[i] = val_dist(gen);
    }
    return d;
}

static FloatTestData make_float_data(size_t nnz, size_t x_size, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
    std::uniform_int_distribution<uint32_t> col_dist(0, static_cast<uint32_t>(x_size - 1));

    FloatTestData d;
    d.nnz = nnz;
    d.vals.resize(nnz);
    d.cols.resize(nnz);
    d.x.resize(x_size);

    for (size_t i = 0; i < nnz; i++) {
        d.vals[i] = val_dist(gen);
        d.cols[i] = col_dist(gen);
    }
    for (size_t i = 0; i < x_size; i++) {
        d.x[i] = val_dist(gen);
    }
    return d;
}

// ============================================================================
// Test sizes — exercises every tail path:
//   Empty(0), single(1), sub-register(2,3,5,7), exact multiples(4,8,16),
//   tail handling(9,13,17,31,33,63,65), larger(100,256,1000,1024)
// ============================================================================

static const std::vector<size_t> TEST_SIZES = {0,  1,  2,  3,  4,  5,   7,   8,   9,   13,  15,  16,   17,  31,
                                               32, 33, 63, 64, 65, 100, 128, 255, 256, 257, 512, 1000, 1024};

constexpr size_t X_SIZE = 2048;

// ============================================================================
// Tolerances
//
// FMA (single rounding) vs scalar mul+add (double rounding) produces
// legitimate differences. Accumulated over many elements these grow.
// ============================================================================

static double double_tol(double expected) { return std::abs(expected) * 1e-10 + 1e-14; }

static double float_tol(float expected) { return std::abs(expected) * 1e-4 + 1e-5; }

// ============================================================================
// SCALAR TESTS — always run on every platform
// ============================================================================

class ScalarDoubleTest : public ::testing::TestWithParam<size_t> {};
class ScalarFloatTest : public ::testing::TestWithParam<size_t> {};

TEST_P(ScalarDoubleTest, MatchesReference) {
    size_t nnz = GetParam();
    auto d = make_double_data(nnz, X_SIZE);
    double expected = reference_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    double actual = sparse_dot_double_scalar(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, double_tol(expected)) << "nnz=" << nnz;
}

TEST_P(ScalarFloatTest, MatchesReference) {
    size_t nnz = GetParam();
    auto d = make_float_data(nnz, X_SIZE);
    float expected = reference_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    float actual = sparse_dot_float_scalar(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, float_tol(expected)) << "nnz=" << nnz;
}

INSTANTIATE_TEST_SUITE_P(Scalar, ScalarDoubleTest, ::testing::ValuesIn(TEST_SIZES));
INSTANTIATE_TEST_SUITE_P(Scalar, ScalarFloatTest, ::testing::ValuesIn(TEST_SIZES));

// ============================================================================
// X86 SIMD TESTS — compiled only on x86, runtime skip if ISA unavailable
// ============================================================================
#if defined(SPIRA_ARCH_X86)

// ---- SSE ----

class SSEDoubleTest : public ::testing::TestWithParam<size_t> {};
class SSEFloatTest : public ::testing::TestWithParam<size_t> {};

TEST_P(SSEDoubleTest, MatchesReference) {
    if (!get_cpu().sse42)
        GTEST_SKIP() << "SSE4.2 not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_double_data(nnz, X_SIZE);
    double expected = reference_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    double actual = sparse_dot_double_sse(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, double_tol(expected)) << "nnz=" << nnz;
}

TEST_P(SSEFloatTest, MatchesReference) {
    if (!get_cpu().sse42)
        GTEST_SKIP() << "SSE4.2 not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_float_data(nnz, X_SIZE);
    float expected = reference_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    float actual = sparse_dot_float_sse(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, float_tol(expected)) << "nnz=" << nnz;
}

INSTANTIATE_TEST_SUITE_P(SSE, SSEDoubleTest, ::testing::ValuesIn(TEST_SIZES));
INSTANTIATE_TEST_SUITE_P(SSE, SSEFloatTest, ::testing::ValuesIn(TEST_SIZES));

// ---- AVX2 ----

class AVX2DoubleTest : public ::testing::TestWithParam<size_t> {};
class AVX2FloatTest : public ::testing::TestWithParam<size_t> {};

TEST_P(AVX2DoubleTest, MatchesReference) {
    if (!get_cpu().avx2 || !get_cpu().fma)
        GTEST_SKIP() << "AVX2+FMA not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_double_data(nnz, X_SIZE);
    double expected = reference_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    double actual = sparse_dot_double_avx(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, double_tol(expected)) << "nnz=" << nnz;
}

TEST_P(AVX2FloatTest, MatchesReference) {
    if (!get_cpu().avx2 || !get_cpu().fma)
        GTEST_SKIP() << "AVX2+FMA not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_float_data(nnz, X_SIZE);
    float expected = reference_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    float actual = sparse_dot_float_avx(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, float_tol(expected)) << "nnz=" << nnz;
}

INSTANTIATE_TEST_SUITE_P(AVX2, AVX2DoubleTest, ::testing::ValuesIn(TEST_SIZES));
INSTANTIATE_TEST_SUITE_P(AVX2, AVX2FloatTest, ::testing::ValuesIn(TEST_SIZES));

// ---- AVX-512 ----

class AVX512DoubleTest : public ::testing::TestWithParam<size_t> {};
class AVX512FloatTest : public ::testing::TestWithParam<size_t> {};

TEST_P(AVX512DoubleTest, MatchesReference) {
    if (!get_cpu().avx512f)
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_double_data(nnz, X_SIZE);
    double expected = reference_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    double actual = sparse_dot_double_avx512(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, double_tol(expected)) << "nnz=" << nnz;
}

TEST_P(AVX512FloatTest, MatchesReference) {
    if (!get_cpu().avx512f)
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_float_data(nnz, X_SIZE);
    float expected = reference_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    float actual = sparse_dot_float_avx512(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, float_tol(expected)) << "nnz=" << nnz;
}

INSTANTIATE_TEST_SUITE_P(AVX512, AVX512DoubleTest, ::testing::ValuesIn(TEST_SIZES));
INSTANTIATE_TEST_SUITE_P(AVX512, AVX512FloatTest, ::testing::ValuesIn(TEST_SIZES));

#endif // SPIRA_ARCH_X86

// ============================================================================
// ARM NEON TESTS — compiled only on ARM, runtime skip if NEON unavailable
// ============================================================================
#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)

class NEONDoubleTest : public ::testing::TestWithParam<size_t> {};
class NEONFloatTest : public ::testing::TestWithParam<size_t> {};

TEST_P(NEONDoubleTest, MatchesReference) {
    if (!get_cpu().neon)
        GTEST_SKIP() << "NEON not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_double_data(nnz, X_SIZE);
    double expected = reference_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    double actual = sparse_dot_double_neon(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, double_tol(expected)) << "nnz=" << nnz;
}

TEST_P(NEONFloatTest, MatchesReference) {
    if (!get_cpu().neon)
        GTEST_SKIP() << "NEON not supported on this CPU";
    size_t nnz = GetParam();
    auto d = make_float_data(nnz, X_SIZE);
    float expected = reference_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    float actual = sparse_dot_float_neon(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, float_tol(expected)) << "nnz=" << nnz;
}

INSTANTIATE_TEST_SUITE_P(NEON, NEONDoubleTest, ::testing::ValuesIn(TEST_SIZES));
INSTANTIATE_TEST_SUITE_P(NEON, NEONFloatTest, ::testing::ValuesIn(TEST_SIZES));

#endif // SPIRA_ARCH_ARM64 || SPIRA_ARCH_ARM32

// ============================================================================
// DISPATCH TESTS — call through function pointers, works on every platform
// ============================================================================

class DispatchDoubleTest : public ::testing::TestWithParam<size_t> {};
class DispatchFloatTest : public ::testing::TestWithParam<size_t> {};

TEST_P(DispatchDoubleTest, MatchesReference) {
    size_t nnz = GetParam();
    auto d = make_double_data(nnz, X_SIZE);
    double expected = reference_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    double actual = spira::kernel::sparse_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, double_tol(expected)) << "nnz=" << nnz;
}

TEST_P(DispatchFloatTest, MatchesReference) {
    size_t nnz = GetParam();
    auto d = make_float_data(nnz, X_SIZE);
    float expected = reference_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    float actual = spira::kernel::sparse_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(expected, actual, float_tol(expected)) << "nnz=" << nnz;
}

INSTANTIATE_TEST_SUITE_P(Dispatch, DispatchDoubleTest, ::testing::ValuesIn(TEST_SIZES));
INSTANTIATE_TEST_SUITE_P(Dispatch, DispatchFloatTest, ::testing::ValuesIn(TEST_SIZES));

// ============================================================================
// KNOWN VALUES — hand-computed, works on every platform
// ============================================================================

TEST(KnownValues, DoubleSimple) {
    double vals[] = {1.0, 2.0, 3.0};
    uint32_t cols[] = {0, 2, 1};
    double x[] = {10.0, 20.0, 30.0};
    EXPECT_DOUBLE_EQ(130.0, sparse_dot_double_scalar(vals, cols, x, 3, 3));
    EXPECT_DOUBLE_EQ(130.0, spira::kernel::sparse_dot_double(vals, cols, x, 3, 3));
}

TEST(KnownValues, FloatSimple) {
    float vals[] = {1.0f, 2.0f, 3.0f};
    uint32_t cols[] = {0, 2, 1};
    float x[] = {10.0f, 20.0f, 30.0f};
    EXPECT_FLOAT_EQ(130.0f, sparse_dot_float_scalar(vals, cols, x, 3, 3));
    EXPECT_FLOAT_EQ(130.0f, spira::kernel::sparse_dot_float(vals, cols, x, 3, 3));
}

TEST(KnownValues, AllZeroVals) {
    double vals[] = {0.0, 0.0, 0.0, 0.0};
    uint32_t cols[] = {0, 1, 2, 3};
    double x[] = {1.0, 2.0, 3.0, 4.0};
    EXPECT_DOUBLE_EQ(0.0, spira::kernel::sparse_dot_double(vals, cols, x, 4, 4));
}

TEST(KnownValues, AllZeroX) {
    double vals[] = {1.0, 2.0, 3.0, 4.0};
    uint32_t cols[] = {0, 1, 2, 3};
    double x[] = {0.0, 0.0, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(0.0, spira::kernel::sparse_dot_double(vals, cols, x, 4, 4));
}

TEST(KnownValues, SingleElement) {
    double vals[] = {7.0};
    uint32_t cols[] = {3};
    double x[] = {0.0, 0.0, 0.0, 5.0};
    EXPECT_DOUBLE_EQ(35.0, spira::kernel::sparse_dot_double(vals, cols, x, 1, 4));
}

TEST(KnownValues, EmptyInput) {
    EXPECT_DOUBLE_EQ(0.0, spira::kernel::sparse_dot_double(nullptr, nullptr, nullptr, 0, 0));
    EXPECT_FLOAT_EQ(0.0f, spira::kernel::sparse_dot_float(nullptr, nullptr, nullptr, 0, 0));
}

TEST(KnownValues, RepeatedColumnIndex) {
    double vals[] = {1.0, 2.0, 3.0};
    uint32_t cols[] = {0, 0, 0};
    double x[] = {5.0};
    EXPECT_DOUBLE_EQ(30.0, spira::kernel::sparse_dot_double(vals, cols, x, 3, 1));
}

TEST(KnownValues, NegativeValues) {
    double vals[] = {-1.0, 2.0, -3.0};
    uint32_t cols[] = {0, 1, 2};
    double x[] = {4.0, -5.0, 6.0};
    EXPECT_DOUBLE_EQ(-32.0, spira::kernel::sparse_dot_double(vals, cols, x, 3, 3));
}

TEST(KnownValues, ExactlyTwoDoubles) {
    double vals[] = {2.0, 3.0};
    uint32_t cols[] = {1, 0};
    double x[] = {10.0, 20.0};
    EXPECT_DOUBLE_EQ(70.0, spira::kernel::sparse_dot_double(vals, cols, x, 2, 2));
}

TEST(KnownValues, ExactlyFourDoubles) {
    double vals[] = {1.0, 2.0, 3.0, 4.0};
    uint32_t cols[] = {0, 1, 2, 3};
    double x[] = {10.0, 20.0, 30.0, 40.0};
    EXPECT_DOUBLE_EQ(300.0, spira::kernel::sparse_dot_double(vals, cols, x, 4, 4));
}

TEST(KnownValues, ExactlyEightFloats) {
    float vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t cols[] = {0, 1, 2, 3, 4, 5, 6, 7};
    float x[] = {1, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_FLOAT_EQ(36.0f, spira::kernel::sparse_dot_float(vals, cols, x, 8, 8));
}

// ============================================================================
// CONSISTENCY — all available implementations agree (platform-adaptive)
// ============================================================================

TEST(Consistency, AllDoubleImplementationsAgree) {
    auto d = make_double_data(137, X_SIZE, 99);
    double ref = reference_dot_double(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    double tol = double_tol(ref);

    double scal = sparse_dot_double_scalar(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(ref, scal, tol) << "scalar";

#if defined(SPIRA_ARCH_X86)
    if (get_cpu().sse42) {
        double sse = sparse_dot_double_sse(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, sse, tol) << "SSE";
    }
    if (get_cpu().avx2 && get_cpu().fma) {
        double avx2 = sparse_dot_double_avx(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, avx2, tol) << "AVX2";
    }
    if (get_cpu().avx512f) {
        double a512 = sparse_dot_double_avx512(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, a512, tol) << "AVX-512";
    }
#endif

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
    if (get_cpu().neon) {
        double neon = sparse_dot_double_neon(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, neon, tol) << "NEON";
    }
#endif
}

TEST(Consistency, AllFloatImplementationsAgree) {
    auto d = make_float_data(137, X_SIZE, 99);
    float ref = reference_dot_float(d.vals.data(), d.cols.data(), d.x.data(), d.nnz);
    float tol = float_tol(ref);

    float scal = sparse_dot_float_scalar(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
    EXPECT_NEAR(ref, scal, tol) << "scalar";

#if defined(SPIRA_ARCH_X86)
    if (get_cpu().sse42) {
        float sse = sparse_dot_float_sse(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, sse, tol) << "SSE";
    }
    if (get_cpu().avx2 && get_cpu().fma) {
        float avx2 = sparse_dot_float_avx(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, avx2, tol) << "AVX2";
    }
    if (get_cpu().avx512f) {
        float a512 = sparse_dot_float_avx512(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, a512, tol) << "AVX-512";
    }
#endif

#if defined(SPIRA_ARCH_ARM64) || defined(SPIRA_ARCH_ARM32)
    if (get_cpu().neon) {
        float neon = sparse_dot_float_neon(d.vals.data(), d.cols.data(), d.x.data(), d.nnz, d.x.size());
        EXPECT_NEAR(ref, neon, tol) << "NEON";
    }
#endif
}

// ============================================================================
// DISPATCH POINTER VALIDITY — works on every platform
// ============================================================================

TEST(DispatchInit, PointersNotNull) {
    EXPECT_NE(nullptr, spira::kernel::sparse_dot_double);
    EXPECT_NE(nullptr, spira::kernel::sparse_dot_float);
}

// ============================================================================
// PLATFORM INFO — not a test, just prints what ISA was selected
// ============================================================================

TEST(PlatformInfo, PrintDetectedFeatures) {
    auto &cpu = get_cpu();
    std::cout << "\n=== CPU Features Detected ===\n";

#if defined(SPIRA_ARCH_X86)
    std::cout << "Architecture: x86\n";
    std::cout << "  SSE2:      " << (cpu.sse2 ? "yes" : "no") << "\n";
    std::cout << "  SSE4.2:    " << (cpu.sse42 ? "yes" : "no") << "\n";
    std::cout << "  AVX:       " << (cpu.avx ? "yes" : "no") << "\n";
    std::cout << "  AVX2:      " << (cpu.avx2 ? "yes" : "no") << "\n";
    std::cout << "  FMA:       " << (cpu.fma ? "yes" : "no") << "\n";
    std::cout << "  AVX-512F:  " << (cpu.avx512f ? "yes" : "no") << "\n";
    std::cout << "  AVX-512BW: " << (cpu.avx512bw ? "yes" : "no") << "\n";
    std::cout << "  AVX-512VL: " << (cpu.avx512vl ? "yes" : "no") << "\n";
    std::cout << "  AVX-512DQ: " << (cpu.avx512dq ? "yes" : "no") << "\n";

    if (cpu.avx512f)
        std::cout << "  Dispatch: AVX-512\n";
    else if (cpu.avx2 && cpu.fma)
        std::cout << "  Dispatch: AVX2+FMA\n";
    else if (cpu.sse42)
        std::cout << "  Dispatch: SSE4.2\n";
    else
        std::cout << "  Dispatch: Scalar\n";
#elif defined(SPIRA_ARCH_ARM64)
    std::cout << "Architecture: ARM64\n";
    std::cout << "  NEON: " << (cpu.neon ? "yes" : "no") << "\n";
    std::cout << "  SVE:  " << (cpu.sve ? "yes" : "no") << "\n";
    std::cout << "  SVE2: " << (cpu.sve2 ? "yes" : "no") << "\n";

    if (cpu.neon)
        std::cout << "  Dispatch: NEON\n";
    else
        std::cout << "  Dispatch: Scalar\n";
#elif defined(SPIRA_ARCH_ARM32)
    std::cout << "Architecture: ARM32\n";
    std::cout << "  NEON: " << (cpu.neon ? "yes" : "no") << "\n";

    if (cpu.neon)
        std::cout << "  Dispatch: NEON\n";
    else
        std::cout << "  Dispatch: Scalar\n";
#else
    std::cout << "Architecture: Unknown\n";
    std::cout << "  Dispatch: Scalar\n";
#endif

    std::cout << "=============================\n";
}
