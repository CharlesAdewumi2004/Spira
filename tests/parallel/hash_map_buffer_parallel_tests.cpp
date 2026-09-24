// tests/parallel/hash_map_buffer_parallel_tests.cpp
//
// Exercises parallel_matrix with hash_map_buffer and the SoA layout — the
// matrix type bench/spira_bench.cpp uses, and the only parallel coverage of
// the SoA/double SIMD SpMV path. When the buffer's sort_and_dedup() was
// missing, lock() here recursed until the stack ran out (-O0) or spun forever
// after tail-call optimisation (-O2).
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <spira/parallel/algorithms/spmv.hpp>
#include <spira/parallel/parallel_matrix.hpp>

using SoA = spira::layout::tags::soa_tag;
using AoS = spira::layout::tags::aos_tag;
using BufTag = spira::buffer::tags::hash_map_buffer;

// Mirrors the `PM` alias in bench/spira_bench.cpp.
template <class LayoutTag>
using PM = spira::parallel::parallel_matrix<LayoutTag, uint32_t, double, BufTag>;

namespace
{
    struct Triple
    {
        std::size_t row;
        uint32_t    col;
        double      val;
    };

    // Deterministic banded fill: 3 entries per row, columns r, (r+1)%n, (r+2)%n.
    std::vector<Triple> banded(std::size_t n)
    {
        std::vector<Triple> out;
        out.reserve(n * 3);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t k = 0; k < 3; ++k)
                out.push_back({r, static_cast<uint32_t>((r + k) % n),
                               static_cast<double>(r + 1) + 0.5 * static_cast<double>(k)});
        return out;
    }

    template <class LayoutTag>
    void fill_and_lock(PM<LayoutTag> &mat, const std::vector<Triple> &triples)
    {
        for (const auto &t : triples)
            mat.insert(t.row, t.col, t.val);
        mat.lock();
    }

    std::vector<double> dense_spmv(std::size_t n,
                                   const std::vector<Triple> &triples,
                                   const std::vector<double> &x)
    {
        std::vector<double> y(n, 0.0);
        for (const auto &t : triples)
            y[t.row] += t.val * x[t.col];
        return y;
    }
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Lock / read-back
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelHashMapBuffer, LockCompletesAndReadsBack_SoA)
{
    constexpr std::size_t N = 64;
    PM<SoA> mat(N, N, 4);
    const auto triples = banded(N);
    fill_and_lock(mat, triples);

    EXPECT_EQ(mat.nnz(), triples.size());
    for (const auto &t : triples)
        EXPECT_DOUBLE_EQ(mat.get(t.row, t.col), t.val)
            << "at (" << t.row << ", " << t.col << ")";
}

TEST(ParallelHashMapBuffer, LockCompletesAndReadsBack_AoS)
{
    constexpr std::size_t N = 64;
    PM<AoS> mat(N, N, 4);
    const auto triples = banded(N);
    fill_and_lock(mat, triples);

    EXPECT_EQ(mat.nnz(), triples.size());
    for (const auto &t : triples)
        EXPECT_DOUBLE_EQ(mat.get(t.row, t.col), t.val);
}

TEST(ParallelHashMapBuffer, ParallelFillThenLock)
{
    constexpr std::size_t N = 64;
    PM<SoA> mat(N, N, 4);

    // Each worker fills only the rows it owns — the bench's assembly path.
    mat.parallel_fill([](auto &rows, std::size_t r_start, std::size_t r_end, std::size_t)
    {
        for (std::size_t r = r_start; r < r_end; ++r)
            rows[r - r_start].insert(static_cast<uint32_t>(r % N), 1.0);
    });
    mat.lock();

    EXPECT_EQ(mat.nnz(), N);
    for (std::size_t r = 0; r < N; ++r)
        EXPECT_DOUBLE_EQ(mat.get(r, static_cast<uint32_t>(r % N)), 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// SpMV — SoA dispatches to the SIMD kernel, AoS to the scalar loop
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelHashMapBuffer, SpmvMatchesDenseReference_SoA)
{
    constexpr std::size_t N = 64;
    PM<SoA> mat(N, N, 4);
    const auto triples = banded(N);
    fill_and_lock(mat, triples);

    std::vector<double> x(N), y(N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = 0.25 * static_cast<double>(i) + 1.0;

    spira::parallel::algorithms::spmv(mat, x, y);

    const auto expected = dense_spmv(N, triples, x);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(y[i], expected[i]) << "row " << i;
}

TEST(ParallelHashMapBuffer, SpmvMatchesDenseReference_AoS)
{
    constexpr std::size_t N = 64;
    PM<AoS> mat(N, N, 4);
    const auto triples = banded(N);
    fill_and_lock(mat, triples);

    std::vector<double> x(N), y(N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        x[i] = 0.25 * static_cast<double>(i) + 1.0;

    spira::parallel::algorithms::spmv(mat, x, y);

    const auto expected = dense_spmv(N, triples, x);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(y[i], expected[i]) << "row " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Incremental update cycle — the bench InsertFixture path
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelHashMapBuffer, ReopenInsertRelockMerges)
{
    constexpr std::size_t N = 64;
    PM<SoA> mat(N, N, 4);
    const auto triples = banded(N);
    fill_and_lock(mat, triples);
    const std::size_t base_nnz = mat.nnz();

    mat.open();
    mat.insert(5, 40u, 123.0);  // new column in row 5
    mat.insert(0, 0u, 777.0);   // overwrite an existing entry
    mat.lock();

    EXPECT_EQ(mat.nnz(), base_nnz + 1);
    EXPECT_DOUBLE_EQ(mat.get(5, 40u), 123.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 0u), 777.0);
}

TEST(ParallelHashMapBuffer, ZeroInsertDeletesCommittedEntry)
{
    constexpr std::size_t N = 64;
    PM<SoA> mat(N, N, 4);
    const auto triples = banded(N);
    fill_and_lock(mat, triples);
    const std::size_t base_nnz = mat.nnz();

    ASSERT_TRUE(mat.contains(5, 5u));

    mat.open();
    mat.insert(5, 5u, 0.0); // zero write = deletion
    mat.lock();

    EXPECT_EQ(mat.nnz(), base_nnz - 1);
    EXPECT_FALSE(mat.contains(5, 5u));
    EXPECT_DOUBLE_EQ(mat.get(5, 5u), 0.0);
}
