#include <benchmark/benchmark.h>
#include <spira/spira.hpp>

#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Common workload parameters — kept identical across all stages
// ---------------------------------------------------------------------------
// Matrix sizes (N×N):  1000, 10000, 50000, 100000
// Non-zeros per row:   4, 16, 64, 256
// Column patterns:     strided (evenly spaced), band (near diagonal), random
// Value type:          double
// RNG seed:            42 (deterministic values only — column placement is
//                          deterministic by construction, not random)
// ---------------------------------------------------------------------------

static constexpr unsigned SEED = 42;

using AoS = spira::layout::tags::aos_tag;
using SoA = spira::layout::tags::soa_tag;

// ---- column placement strategies ------------------------------------------

static std::vector<uint32_t> strided_cols(size_t N, int nnz) {
    std::vector<uint32_t> cols(nnz);
    size_t stride = N / static_cast<size_t>(nnz);
    if (stride == 0) stride = 1;
    for (int k = 0; k < nnz; ++k)
        cols[k] = static_cast<uint32_t>((static_cast<size_t>(k) * stride) % N);
    return cols;
}

static std::vector<uint32_t> band_cols(size_t row, size_t N, int nnz) {
    std::vector<uint32_t> cols(nnz);
    int half = nnz / 2;
    for (int k = 0; k < nnz; ++k) {
        auto c = static_cast<int64_t>(row) + (k - half);
        cols[k] = static_cast<uint32_t>(((c % static_cast<int64_t>(N)) + static_cast<int64_t>(N)) % static_cast<int64_t>(N));
    }
    return cols;
}

static std::vector<uint32_t> random_cols(size_t row, size_t N, int nnz) {
    std::mt19937 rng(SEED ^ static_cast<unsigned>(row));
    std::uniform_int_distribution<uint32_t> col_dist(0, static_cast<uint32_t>(N - 1));
    std::vector<uint32_t> cols(nnz);
    for (int k = 0; k < nnz; ++k)
        cols[k] = col_dist(rng);
    return cols;
}

// ---- fill helpers ---------------------------------------------------------

template <class Mat>
static void fill_strided(Mat &mat, size_t N, int nnz, std::mt19937 &rng) {
    std::uniform_real_distribution<double> val_dist(0.0, 1.0);
    auto cols = strided_cols(N, nnz);
    for (size_t r = 0; r < N; ++r) {
        for (int k = 0; k < nnz; ++k) {
            mat.insert(static_cast<uint32_t>(r), cols[k], val_dist(rng));
        }
    }
}

template <class Mat>
static void fill_band(Mat &mat, size_t N, int nnz, std::mt19937 &rng) {
    std::uniform_real_distribution<double> val_dist(0.0, 1.0);
    for (size_t r = 0; r < N; ++r) {
        auto cols = band_cols(r, N, nnz);
        for (int k = 0; k < nnz; ++k) {
            mat.insert(static_cast<uint32_t>(r), cols[k], val_dist(rng));
        }
    }
}

template <class Mat>
static void fill_random(Mat &mat, size_t N, int nnz, std::mt19937 &rng) {
    std::uniform_real_distribution<double> val_dist(0.0, 1.0);
    for (size_t r = 0; r < N; ++r) {
        auto cols = random_cols(r, N, nnz);
        for (int k = 0; k < nnz; ++k) {
            mat.insert(static_cast<uint32_t>(r), cols[k], val_dist(rng));
        }
    }
}

template <class V>
static std::vector<V> random_vector(size_t n, std::mt19937 &rng) {
    std::uniform_real_distribution<V> dist(V(0), V(1));
    std::vector<V> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

// ---- benchmark argument encoding ------------------------------------------
// range(0) = N  (matrix dimension)
// range(1) = nnz per row

static void AllSizesAndDensities(benchmark::internal::Benchmark *b) {
    for (int N : {1000, 10000, 50000, 100000})
        for (int nnz : {4, 16, 64, 256})
            b->Args({N, nnz});
}

// ===========================================================================
//  Insertion benchmarks — strided columns
// ===========================================================================

template <class LayoutTag>
static void BM_Insertion_Strided(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();
        std::mt19937 rng(SEED);
        state.ResumeTiming();

        spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
        mat.set_mode(spira::mode::matrix_mode::insert_heavy);
        fill_strided(mat, N, nnz, rng);

        benchmark::DoNotOptimize(&mat);
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz);
}

BENCHMARK(BM_Insertion_Strided<AoS>)
    ->Name("Insertion_Strided/AoS")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Insertion_Strided<SoA>)
    ->Name("Insertion_Strided/SoA")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  Insertion benchmarks — band columns
// ===========================================================================

template <class LayoutTag>
static void BM_Insertion_Band(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();
        std::mt19937 rng(SEED);
        state.ResumeTiming();

        spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
        mat.set_mode(spira::mode::matrix_mode::insert_heavy);
        fill_band(mat, N, nnz, rng);

        benchmark::DoNotOptimize(&mat);
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz);
}

BENCHMARK(BM_Insertion_Band<AoS>)
    ->Name("Insertion_Band/AoS")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Insertion_Band<SoA>)
    ->Name("Insertion_Band/SoA")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  Insertion benchmarks — random columns
// ===========================================================================

template <class LayoutTag>
static void BM_Insertion_Random(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();
        std::mt19937 rng(SEED);
        state.ResumeTiming();

        spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
        mat.set_mode(spira::mode::matrix_mode::insert_heavy);
        fill_random(mat, N, nnz, rng);

        benchmark::DoNotOptimize(&mat);
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz);
}

BENCHMARK(BM_Insertion_Random<AoS>)
    ->Name("Insertion_Random/AoS")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Insertion_Random<SoA>)
    ->Name("Insertion_Random/SoA")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  SpMV benchmarks — strided columns
// ===========================================================================

template <class LayoutTag>
static void BM_SpMV_Strided(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    std::mt19937 rng(SEED);
    spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);
    fill_strided(mat, N, nnz, rng);
    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    auto x = random_vector<double>(N, rng);
    std::vector<double> y(N);

    for (auto _ : state) {
        spira::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz * 2);
}

BENCHMARK(BM_SpMV_Strided<AoS>)
    ->Name("SpMV_Strided/AoS/double")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_SpMV_Strided<SoA>)
    ->Name("SpMV_Strided/SoA/double")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  SpMV benchmarks — band columns (double)
// ===========================================================================

template <class LayoutTag>
static void BM_SpMV_Band(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    std::mt19937 rng(SEED);
    spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);
    fill_band(mat, N, nnz, rng);
    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    auto x = random_vector<double>(N, rng);
    std::vector<double> y(N);

    for (auto _ : state) {
        spira::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz * 2);
}

BENCHMARK(BM_SpMV_Band<AoS>)
    ->Name("SpMV_Band/AoS/double")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_SpMV_Band<SoA>)
    ->Name("SpMV_Band/SoA/double")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  SpMV benchmarks — random columns (double)
// ===========================================================================

template <class LayoutTag>
static void BM_SpMV_Random(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    std::mt19937 rng(SEED);
    spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);
    fill_random(mat, N, nnz, rng);
    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    auto x = random_vector<double>(N, rng);
    std::vector<double> y(N);

    for (auto _ : state) {
        spira::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz * 2);
}

BENCHMARK(BM_SpMV_Random<AoS>)
    ->Name("SpMV_Random/AoS/double")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_SpMV_Random<SoA>)
    ->Name("SpMV_Random/SoA/double")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  SpMV benchmarks — strided columns (float) — exercises SIMD float path
// ===========================================================================

template <class LayoutTag>
static void BM_SpMV_Strided_Float(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    std::mt19937 rng(SEED);
    spira::matrix<LayoutTag, uint32_t, float> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);
    std::uniform_real_distribution<float> val_dist(0.0f, 1.0f);
    auto cols = strided_cols(N, nnz);
    for (size_t r = 0; r < N; ++r)
        for (int k = 0; k < nnz; ++k)
            mat.insert(static_cast<uint32_t>(r), cols[k], val_dist(rng));
    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    auto x = random_vector<float>(N, rng);
    std::vector<float> y(N);

    for (auto _ : state) {
        spira::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz * 2);
}

BENCHMARK(BM_SpMV_Strided_Float<AoS>)
    ->Name("SpMV_Strided/AoS/float")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_SpMV_Strided_Float<SoA>)
    ->Name("SpMV_Strided/SoA/float")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  SpMV benchmarks — band columns (float)
// ===========================================================================

template <class LayoutTag>
static void BM_SpMV_Band_Float(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    std::mt19937 rng(SEED);
    spira::matrix<LayoutTag, uint32_t, float> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);
    std::uniform_real_distribution<float> val_dist(0.0f, 1.0f);
    for (size_t r = 0; r < N; ++r) {
        auto cols = band_cols(r, N, nnz);
        for (int k = 0; k < nnz; ++k)
            mat.insert(static_cast<uint32_t>(r), cols[k], val_dist(rng));
    }
    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    auto x = random_vector<float>(N, rng);
    std::vector<float> y(N);

    for (auto _ : state) {
        spira::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz * 2);
}

BENCHMARK(BM_SpMV_Band_Float<AoS>)
    ->Name("SpMV_Band/AoS/float")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_SpMV_Band_Float<SoA>)
    ->Name("SpMV_Band/SoA/float")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  SpMV benchmarks — random columns (float)
// ===========================================================================

template <class LayoutTag>
static void BM_SpMV_Random_Float(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    std::mt19937 rng(SEED);
    spira::matrix<LayoutTag, uint32_t, float> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);
    std::uniform_real_distribution<float> val_dist(0.0f, 1.0f);
    for (size_t r = 0; r < N; ++r) {
        auto cols = random_cols(r, N, nnz);
        for (int k = 0; k < nnz; ++k)
            mat.insert(static_cast<uint32_t>(r), cols[k], val_dist(rng));
    }
    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    auto x = random_vector<float>(N, rng);
    std::vector<float> y(N);

    for (auto _ : state) {
        spira::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(N) * nnz * 2);
}

BENCHMARK(BM_SpMV_Random_Float<AoS>)
    ->Name("SpMV_Random/AoS/float")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_SpMV_Random_Float<SoA>)
    ->Name("SpMV_Random/SoA/float")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================
//  Transition benchmark — cost of set_mode(spmv) + flush()
// ===========================================================================

template <class LayoutTag>
static void BM_Transition_Strided(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();
        std::mt19937 rng(SEED);
        spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
        mat.set_mode(spira::mode::matrix_mode::insert_heavy);
        fill_strided(mat, N, nnz, rng);
        state.ResumeTiming();

        mat.set_mode(spira::mode::matrix_mode::spmv);
        mat.flush();

        benchmark::DoNotOptimize(&mat);
    }
}

BENCHMARK(BM_Transition_Strided<AoS>)
    ->Name("Transition_Strided/AoS")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Transition_Strided<SoA>)
    ->Name("Transition_Strided/SoA")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

template <class LayoutTag>
static void BM_Transition_Band(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();
        std::mt19937 rng(SEED);
        spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
        mat.set_mode(spira::mode::matrix_mode::insert_heavy);
        fill_band(mat, N, nnz, rng);
        state.ResumeTiming();

        mat.set_mode(spira::mode::matrix_mode::spmv);
        mat.flush();

        benchmark::DoNotOptimize(&mat);
    }
}

BENCHMARK(BM_Transition_Band<AoS>)
    ->Name("Transition_Band/AoS")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Transition_Band<SoA>)
    ->Name("Transition_Band/SoA")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

template <class LayoutTag>
static void BM_Transition_Random(benchmark::State &state) {
    const auto N = static_cast<size_t>(state.range(0));
    const auto nnz = static_cast<int>(state.range(1));

    for (auto _ : state) {
        state.PauseTiming();
        std::mt19937 rng(SEED);
        spira::matrix<LayoutTag, uint32_t, double> mat(N, N);
        mat.set_mode(spira::mode::matrix_mode::insert_heavy);
        fill_random(mat, N, nnz, rng);
        state.ResumeTiming();

        mat.set_mode(spira::mode::matrix_mode::spmv);
        mat.flush();

        benchmark::DoNotOptimize(&mat);
    }
}

BENCHMARK(BM_Transition_Random<AoS>)
    ->Name("Transition_Random/AoS")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Transition_Random<SoA>)
    ->Name("Transition_Random/SoA")
    ->Apply(AllSizesAndDensities)
    ->Unit(benchmark::kNanosecond);

// ===========================================================================

BENCHMARK_MAIN();
