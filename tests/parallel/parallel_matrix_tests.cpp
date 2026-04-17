// tests/parallel/parallel_matrix_tests.cpp
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>
#include <numeric>
#include <cmath>

#include <spira/parallel/parallel_matrix.hpp>
#include <spira/parallel/algorithms/spmv.hpp>

using namespace spira;
using namespace spira::parallel;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Simple alias for the default parallel matrix
template <std::size_t N = 2>
using pmat = parallel_matrix<layout::tags::aos_tag,
                             uint32_t, double,
                             buffer::tags::array_buffer<layout::tags::aos_tag>,
                             64,
                             config::lock_policy::compact_preserve,
                             config::insert_policy::direct,
                             256>;

template <std::size_t N = 2>
using pmat_staged = parallel_matrix<layout::tags::aos_tag,
                                    uint32_t, double,
                                    buffer::tags::array_buffer<layout::tags::aos_tag>,
                                    64,
                                    config::lock_policy::compact_preserve,
                                    config::insert_policy::staged,
                                    256>;

// Dense reference SpMV — independent of Spira internals
static std::vector<double> dense_spmv(std::size_t n_rows,
                                      const std::vector<std::tuple<std::size_t, uint32_t, double>> &entries,
                                      const std::vector<double> &x)
{
    std::vector<double> y(n_rows, 0.0);
    for (auto &[r, c, v] : entries)
        y[r] += v * x[static_cast<std::size_t>(c)];
    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelMatrixConstruct, ShapeAndThreadCount)
{
    pmat<> m(8, 10, 2);
    EXPECT_EQ(m.n_rows(), 8u);
    EXPECT_EQ(m.n_cols(), 10u);
    EXPECT_EQ(m.n_threads(), 2u);
    EXPECT_EQ(m.shape(), std::make_pair(std::size_t{8}, std::size_t{10}));
}

TEST(ParallelMatrixConstruct, InitiallyEmpty)
{
    pmat<> m(4, 4, 2);
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.nnz(), 0u);
    EXPECT_TRUE(m.is_open());
    EXPECT_FALSE(m.is_locked());
}

TEST(ParallelMatrixConstruct, SingleThread)
{
    pmat<> m(4, 4, 1);
    EXPECT_EQ(m.n_threads(), 1u);
    m.insert(0, 0u, 1.0);
    m.insert(3, 3u, 2.0);
    m.lock();
    EXPECT_EQ(m.nnz(), 2u);
    EXPECT_DOUBLE_EQ(m.get(0, 0u), 1.0);
    EXPECT_DOUBLE_EQ(m.get(3, 3u), 2.0);
}

TEST(ParallelMatrixConstruct, FourThreads)
{
    pmat<> m(16, 16, 4);
    EXPECT_EQ(m.n_threads(), 4u);
    EXPECT_TRUE(m.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert + query (open mode, before lock)
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelMatrixInsert, ContainsAndGet)
{
    pmat<> m(8, 8, 2);
    m.insert(0, 1u, 1.5);
    m.insert(4, 2u, 2.5);

    EXPECT_TRUE(m.contains(0, 1u));
    EXPECT_DOUBLE_EQ(m.get(0, 1u), 1.5);
    EXPECT_TRUE(m.contains(4, 2u));
    EXPECT_DOUBLE_EQ(m.get(4, 2u), 2.5);
    EXPECT_FALSE(m.contains(0, 2u));
    EXPECT_DOUBLE_EQ(m.get(0, 2u), 0.0);
}

TEST(ParallelMatrixInsert, NnzAndRowNnz)
{
    pmat<> m(8, 8, 2);
    m.insert(0, 0u, 1.0);
    m.insert(0, 1u, 2.0);
    m.insert(7, 3u, 3.0);

    EXPECT_EQ(m.nnz(), 3u);
    EXPECT_EQ(m.row_nnz(0), 2u);
    EXPECT_EQ(m.row_nnz(7), 1u);
    EXPECT_EQ(m.row_nnz(1), 0u);
}

TEST(ParallelMatrixInsert, InsertsSpanAllPartitions)
{
    // 8 rows, 2 threads → rows 0-3 in part 0, rows 4-7 in part 1
    pmat<> m(8, 8, 2);
    for (std::size_t r = 0; r < 8; ++r)
        m.insert(r, static_cast<uint32_t>(r), static_cast<double>(r + 1));

    EXPECT_EQ(m.nnz(), 8u);
    for (std::size_t r = 0; r < 8; ++r)
        EXPECT_DOUBLE_EQ(m.get(r, static_cast<uint32_t>(r)), static_cast<double>(r + 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Lock / open cycle
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelMatrixLock, ModeSwitches)
{
    pmat<> m(4, 4, 2);
    EXPECT_TRUE(m.is_open());
    m.insert(0, 0u, 1.0);
    m.lock();
    EXPECT_TRUE(m.is_locked());
    EXPECT_FALSE(m.is_open());
    m.open();
    EXPECT_TRUE(m.is_open());
    EXPECT_FALSE(m.is_locked());
}

TEST(ParallelMatrixLock, DataPreservedAfterLock)
{
    pmat<> m(8, 8, 2);
    m.insert(1, 2u, 3.14);
    m.insert(5, 6u, 2.72);
    m.lock();

    EXPECT_EQ(m.nnz(), 2u);
    EXPECT_DOUBLE_EQ(m.get(1, 2u), 3.14);
    EXPECT_DOUBLE_EQ(m.get(5, 6u), 2.72);
    EXPECT_FALSE(m.contains(1, 3u));
}

TEST(ParallelMatrixLock, MultipleInsertLockOpenCycles)
{
    pmat<> m(8, 8, 2);

    // cycle 1
    m.insert(0, 0u, 1.0);
    m.insert(4, 4u, 2.0);
    m.lock();
    EXPECT_EQ(m.nnz(), 2u);
    m.open();

    // cycle 2 — insert into different rows
    m.insert(1, 1u, 3.0);
    m.insert(5, 5u, 4.0);
    m.lock();
    EXPECT_EQ(m.nnz(), 4u);
    EXPECT_DOUBLE_EQ(m.get(0, 0u), 1.0);
    EXPECT_DOUBLE_EQ(m.get(1, 1u), 3.0);
    EXPECT_DOUBLE_EQ(m.get(4, 4u), 2.0);
    EXPECT_DOUBLE_EQ(m.get(5, 5u), 4.0);
    m.open();

    // cycle 3 — insert duplicate: buffer wins on collision (replaces old CSR value)
    m.insert(0, 0u, 10.0);
    m.lock();
    EXPECT_DOUBLE_EQ(m.get(0, 0u), 10.0); // buffer value replaces committed CSR value
}

TEST(ParallelMatrixLock, ZeroInsertDeletesCommittedEntry)
{
    // Regression: zero-value inserts used as deletions were silently ignored
    // because sort_and_dedup() stripped them before merge_csr could see them.
    pmat<> m(8, 8, 2);

    // Cycle 1: commit entries.
    m.insert(0, 1u, 5.0);
    m.insert(0, 2u, 6.0);
    m.insert(4, 4u, 9.0);
    m.lock();
    ASSERT_EQ(m.nnz(), 3u);

    // Cycle 2: delete col 1 from row 0, leave col 2 and row 4 intact.
    m.open();
    m.insert(0, 1u, 0.0);
    m.lock();

    EXPECT_EQ(m.nnz(), 2u);
    EXPECT_FALSE(m.contains(0, 1u));
    EXPECT_DOUBLE_EQ(m.get(0, 2u), 6.0);
    EXPECT_DOUBLE_EQ(m.get(4, 4u), 9.0);
}

TEST(ParallelMatrixLock, ClearRemovesPendingBufferInserts)
{
    pmat<> m(8, 8, 2);
    // Insert and lock — data committed to CSR
    m.insert(0, 0u, 1.0);
    m.insert(4, 4u, 2.0);
    m.lock();
    m.open();

    // Insert new pending entries, then clear before locking
    m.insert(1, 1u, 99.0);
    m.insert(5, 5u, 88.0);
    m.clear();

    // Pending buffer inserts are gone; committed CSR history persists
    EXPECT_FALSE(m.contains(1, 1u));
    EXPECT_FALSE(m.contains(5, 5u));
    EXPECT_TRUE(m.contains(0, 0u)); // still in CSR
    EXPECT_TRUE(m.contains(4, 4u)); // still in CSR
}

TEST(ParallelMatrixLock, ClearBeforeLockGivesEmptyMatrix)
{
    pmat<> m(8, 8, 2);
    m.insert(0, 0u, 1.0);
    m.insert(4, 4u, 2.0);
    // clear before any lock — no CSR committed yet
    m.clear();

    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.nnz(), 0u);
    EXPECT_FALSE(m.contains(0, 0u));
}

// ─────────────────────────────────────────────────────────────────────────────
// Staged insert policy — same correctness as direct
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelMatrixStaged, BasicInsertAndLock)
{
    pmat_staged<> m(8, 8, 2);
    m.insert(0, 1u, 1.5);
    m.insert(4, 2u, 2.5);
    m.insert(7, 7u, 3.5);
    m.lock();

    EXPECT_EQ(m.nnz(), 3u);
    EXPECT_DOUBLE_EQ(m.get(0, 1u), 1.5);
    EXPECT_DOUBLE_EQ(m.get(4, 2u), 2.5);
    EXPECT_DOUBLE_EQ(m.get(7, 7u), 3.5);
}

TEST(ParallelMatrixStaged, MatchesDirectPolicy)
{
    // Build same matrix with both policies, compare nnz + all gets
    const std::size_t N = 12;
    pmat<> direct(N, N, 2);
    pmat_staged<> staged(N, N, 2);

    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < 3; ++c)
        {
            auto col = static_cast<uint32_t>((r + c) % N);
            auto val = static_cast<double>(r * 10 + c);
            direct.insert(r, col, val);
            staged.insert(r, col, val);
        }

    direct.lock();
    staged.lock();

    EXPECT_EQ(direct.nnz(), staged.nnz());
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c)
            EXPECT_DOUBLE_EQ(direct.get(r, static_cast<uint32_t>(c)),
                             staged.get(r, static_cast<uint32_t>(c)))
                << "mismatch at (" << r << "," << c << ")";
}

TEST(ParallelMatrixStaged, StagingBufferFlushOnCapacity)
{
    // StagingN = 4, insert more than 4 into one partition to force mid-insert flush
    using small_staged = parallel_matrix<layout::tags::aos_tag,
                                         uint32_t, double,
                                         buffer::tags::array_buffer<layout::tags::aos_tag>,
                                         64,
                                         config::lock_policy::compact_preserve,
                                         config::insert_policy::staged,
                                         4>; // tiny staging buffer
    small_staged m(8, 16, 1);                // single thread → all rows one partition, 16 cols
    for (uint32_t c = 0; c < 10; ++c)
        m.insert(0, c, static_cast<double>(c + 1));

    m.lock();
    EXPECT_EQ(m.row_nnz(0), 10u);
    for (uint32_t c = 0; c < 10; ++c)
        EXPECT_DOUBLE_EQ(m.get(0, c), static_cast<double>(c + 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel SpMV correctness
// ─────────────────────────────────────────────────────────────────────────────

namespace
{
    // Build a small matrix, lock it, run parallel spmv, compare to dense ref
    void run_spmv_test(std::size_t n_rows, std::size_t n_cols, std::size_t n_threads,
                       const std::vector<std::tuple<std::size_t, uint32_t, double>> &entries)
    {
        pmat<> m(n_rows, n_cols, n_threads);
        for (auto &[r, c, v] : entries)
            m.insert(r, c, v);
        m.lock();

        std::vector<double> x(n_cols);
        std::iota(x.begin(), x.end(), 1.0);

        std::vector<double> y(n_rows, 0.0);
        algorithms::spmv(m, x, y);

        auto ref = dense_spmv(n_rows, entries, x);
        for (std::size_t i = 0; i < n_rows; ++i)
            EXPECT_DOUBLE_EQ(y[i], ref[i]) << "row " << i;
    }
}

TEST(ParallelSpmv, DiagonalMatrix2Threads)
{
    std::vector<std::tuple<std::size_t, uint32_t, double>> entries;
    for (std::size_t i = 0; i < 8; ++i)
        entries.emplace_back(i, static_cast<uint32_t>(i), static_cast<double>(i + 1));
    run_spmv_test(8, 8, 2, entries);
}

TEST(ParallelSpmv, DiagonalMatrix4Threads)
{
    std::vector<std::tuple<std::size_t, uint32_t, double>> entries;
    for (std::size_t i = 0; i < 16; ++i)
        entries.emplace_back(i, static_cast<uint32_t>(i), static_cast<double>(i + 1));
    run_spmv_test(16, 16, 4, entries);
}

TEST(ParallelSpmv, DenseSmallMatrix)
{
    // 4×4 fully dense
    std::vector<std::tuple<std::size_t, uint32_t, double>> entries;
    for (std::size_t r = 0; r < 4; ++r)
        for (uint32_t c = 0; c < 4; ++c)
            entries.emplace_back(r, c, static_cast<double>(r * 4 + c + 1));
    run_spmv_test(4, 4, 2, entries);
}

TEST(ParallelSpmv, SparseIrregular)
{
    // irregular sparsity pattern across partitions
    std::vector<std::tuple<std::size_t, uint32_t, double>> entries = {
        {0, 0u, 1.0}, {0, 3u, 2.0}, {2, 1u, 3.0}, {3, 2u, 4.0}, {3, 3u, 5.0}, {5, 0u, 6.0}, {5, 2u, 7.0}, {7, 1u, 8.0}};
    run_spmv_test(8, 4, 2, entries);
}

TEST(ParallelSpmv, SingleThread)
{
    std::vector<std::tuple<std::size_t, uint32_t, double>> entries;
    for (std::size_t i = 0; i < 6; ++i)
        entries.emplace_back(i, static_cast<uint32_t>(i % 4), static_cast<double>(i + 1));
    run_spmv_test(6, 4, 1, entries);
}

TEST(ParallelSpmv, MatchesAcrossThreadCounts)
{
    // Same matrix, different thread counts → same result
    std::vector<std::tuple<std::size_t, uint32_t, double>> entries;
    for (std::size_t r = 0; r < 8; ++r)
        for (uint32_t c = 0; c < 3; ++c)
            entries.emplace_back(r, c, static_cast<double>(r + c + 1));

    std::vector<double> x(8);
    std::iota(x.begin(), x.end(), 1.0);

    auto get_y = [&](std::size_t n_threads)
    {
        pmat<> m(8, 8, n_threads);
        for (auto &[r, c, v] : entries)
            m.insert(r, c, v);
        m.lock();
        std::vector<double> y(8, 0.0);
        algorithms::spmv(m, x, y);
        return y;
    };

    auto y1 = get_y(1);
    auto y2 = get_y(2);
    auto y4 = get_y(4);

    for (std::size_t i = 0; i < 8; ++i)
    {
        EXPECT_DOUBLE_EQ(y1[i], y2[i]) << "1 vs 2 threads at row " << i;
        EXPECT_DOUBLE_EQ(y1[i], y4[i]) << "1 vs 4 threads at row " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming (PAEM) cycle
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelMatrixStreaming, InsertLockSpmvOpenRepeat)
{
    // Simulates the PAEM loop: insert → lock → spmv → open → insert → ...
    const std::size_t N = 8;
    pmat<> m(N, N, 2);
    std::vector<double> x(N, 1.0);
    std::vector<double> y(N, 0.0);

    // Round 1
    for (std::size_t i = 0; i < N; ++i)
        m.insert(i, static_cast<uint32_t>(i), 1.0);
    m.lock();
    algorithms::spmv(m, x, y);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(y[i], 1.0) << "round 1 row " << i;
    m.open();

    // Round 2 — add second diagonal offset by 1 column
    for (std::size_t i = 0; i < N - 1; ++i)
        m.insert(i, static_cast<uint32_t>(i + 1), 2.0);
    m.lock();
    std::fill(y.begin(), y.end(), 0.0);
    algorithms::spmv(m, x, y);
    // row 0: 1*1 + 2*1 = 3, rows 1-6: same, row 7: only diagonal = 1
    for (std::size_t i = 0; i < N - 1; ++i)
        EXPECT_DOUBLE_EQ(y[i], 3.0) << "round 2 row " << i;
    EXPECT_DOUBLE_EQ(y[N - 1], 1.0);
    m.open();
}
