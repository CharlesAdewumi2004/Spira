#include "config.hpp"

template <class V, class Layout>
static void BM_spmv(benchmark::State &state)
{
    using I = std::size_t;

    const std::size_t n_coordinates = static_cast<std::size_t>(state.range(0)); // nnz
    const std::size_t rows = static_cast<std::size_t>(state.range(1));
    const std::size_t cols = static_cast<std::size_t>(state.range(2));

    spira::matrix<Layout, I, V> m(rows, cols);
    m.set_mode(spira::mode::matrix_mode::insert_heavy);
    std::vector<V> x;
    std::vector<V> y(rows, V{});

    x.reserve(cols);

        std::mt19937_64 rng(12345);
        std::uniform_int_distribution<I> row_dist(0, rows - 1);
        std::uniform_int_distribution<I> col_dist(0, cols - 1);
        std::uniform_real_distribution<double> num_dist(minMaxtrixValue, maxMatrixValue);

        for (std::size_t i = 0; i < cols; ++i)
        {
            x.push_back(static_cast<V>(num_dist(rng)));
        }

    for (std::size_t k = 0; k < n_coordinates; ++k)
    {
        m.add(row_dist(rng), col_dist(rng), static_cast<V>(num_dist(rng)));
    }

    m.set_mode(spira::mode::matrix_mode::spmv);
    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(y.data());

    for (auto _ : state)
    {
        spira::algorithms::spmv(m, x, y);
        benchmark::ClobberMemory();
    }
    const std::size_t bytes_per_nnz =
        sizeof(I)   
        + sizeof(V)  
        + sizeof(V); 
    const std::size_t bytes_per_spmv_y =
        rows * sizeof(V);

    const auto iters = static_cast<int64_t>(state.iterations());
    const auto nnz = static_cast<int64_t>(n_coordinates);

    state.SetItemsProcessed(iters * nnz);

    const auto bytes_per_spmv_total =
        static_cast<int64_t>(n_coordinates) * static_cast<int64_t>(bytes_per_nnz) + static_cast<int64_t>(bytes_per_spmv_y);

    state.SetBytesProcessed(iters * bytes_per_spmv_total);
}


// AOS problemSizeScaling
BENCHMARK_TEMPLATE(BM_spmv, float, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-spmv-float-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_spmv, double, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-spmv-double-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-spmv-complex-problemSizeScaling");

// SOA problemSizeScaling
BENCHMARK_TEMPLATE(BM_spmv, float, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-spmv-float-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_spmv, double, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-spmv-double-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-spmv-complex-problemSizeScaling");

// AOS sparsityScaling
BENCHMARK_TEMPLATE(BM_spmv, float, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-spmv-float-sparsityScaling");
BENCHMARK_TEMPLATE(BM_spmv, double, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-spmv-double-sparsityScaling");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-spmv-complex-sparsityScaling");

// SOA sparsityScaling
BENCHMARK_TEMPLATE(BM_spmv, float, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-spmv-float-sparsityScaling");
BENCHMARK_TEMPLATE(BM_spmv, double, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-spmv-double-sparsityScaling");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-spmv-complex-sparsityScaling");