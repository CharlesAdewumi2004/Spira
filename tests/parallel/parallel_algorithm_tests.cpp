// tests/parallel/parallel_algorithm_tests.cpp
#include <gtest/gtest.h>
#include <spira/spira.hpp>

#include <cstddef>
#include <vector>
#include <numeric>
#include <cmath>
#include <tuple>

using namespace spira;
using namespace spira::parallel;
namespace palg = spira::parallel::algorithms;

// ─────────────────────────────────────────────────────────────────────────────
// Test matrix type (AOS, uint32_t, double, 2 threads)
// ─────────────────────────────────────────────────────────────────────────────

using pmat = parallel_matrix<layout::tags::aos_tag,
                              uint32_t, double,
                              buffer::tags::array_buffer<layout::tags::aos_tag>,
                              64,
                              config::lock_policy::compact_preserve,
                              config::insert_policy::direct,
                              256>;

using entry = std::tuple<std::size_t, uint32_t, double>;

static pmat build_locked(std::size_t n_rows, std::size_t n_cols,
                          std::size_t n_threads,
                          const std::vector<entry> &entries)
{
    pmat m(n_rows, n_cols, n_threads);
    for (auto &[r, c, v] : entries)
        m.insert(r, c, v);
    m.lock();
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// accumulate
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelAccumulate, SingleRowSum)
{
    auto m = build_locked(4, 4, 2, {
        {0, 0u, 1.0}, {0, 1u, 2.0}, {0, 2u, 3.0},
        {1, 0u, 10.0}
    });

    EXPECT_DOUBLE_EQ(palg::accumulate(m, 0), 6.0);
    EXPECT_DOUBLE_EQ(palg::accumulate(m, 1), 10.0);
    EXPECT_DOUBLE_EQ(palg::accumulate(m, 2), 0.0);
    EXPECT_DOUBLE_EQ(palg::accumulate(m, 3), 0.0);
}

TEST(ParallelAccumulate, AllRowsParallel)
{
    std::vector<entry> entries;
    for (std::size_t r = 0; r < 8; ++r)
        for (uint32_t c = 0; c < 4; ++c)
            entries.emplace_back(r, c, static_cast<double>(r + 1));

    auto m = build_locked(8, 4, 2, entries);
    auto result = palg::accumulate(m);

    ASSERT_EQ(result.size(), 8u);
    for (std::size_t r = 0; r < 8; ++r)
        EXPECT_DOUBLE_EQ(result[r], 4.0 * static_cast<double>(r + 1))
            << "row " << r;
}

TEST(ParallelAccumulate, EmptyRowsAreZero)
{
    auto m = build_locked(6, 6, 2, {{2, 2u, 5.0}});
    auto result = palg::accumulate(m);

    for (std::size_t r = 0; r < 6; ++r)
    {
        if (r == 2)
            EXPECT_DOUBLE_EQ(result[r], 5.0);
        else
            EXPECT_DOUBLE_EQ(result[r], 0.0) << "row " << r << " should be zero";
    }
}

TEST(ParallelAccumulate, MatchesAcrossThreadCounts)
{
    std::vector<entry> entries;
    for (std::size_t r = 0; r < 8; ++r)
        for (uint32_t c = 0; c < 3; ++c)
            entries.emplace_back(r, c, static_cast<double>(c + 1));

    auto m1 = build_locked(8, 4, 1, entries);
    auto m2 = build_locked(8, 4, 2, entries);
    auto m4 = build_locked(8, 4, 4, entries);

    auto r1 = palg::accumulate(m1);
    auto r2 = palg::accumulate(m2);
    auto r4 = palg::accumulate(m4);

    for (std::size_t i = 0; i < 8; ++i)
    {
        EXPECT_DOUBLE_EQ(r1[i], r2[i]) << "1 vs 2 threads row " << i;
        EXPECT_DOUBLE_EQ(r1[i], r4[i]) << "1 vs 4 threads row " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// scalars
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelScalars, MultiplicationScalerCopy)
{
    auto A = build_locked(8, 8, 2, {
        {0, 0u, 2.0}, {1, 1u, 4.0},
        {4, 2u, 6.0}, {7, 3u, 8.0}
    });

    pmat out(8, 8, 2);
    palg::multiplication_scaler(A, out, 3.0);
    // out is now locked

    EXPECT_DOUBLE_EQ(out.get(0, 0u), 6.0);
    EXPECT_DOUBLE_EQ(out.get(1, 1u), 12.0);
    EXPECT_DOUBLE_EQ(out.get(4, 2u), 18.0);
    EXPECT_DOUBLE_EQ(out.get(7, 3u), 24.0);
    // unchanged rows stay zero
    EXPECT_DOUBLE_EQ(out.get(2, 0u), 0.0);
}

TEST(ParallelScalars, DivisionScalerCopy)
{
    auto A = build_locked(8, 8, 2, {
        {0, 0u, 6.0}, {3, 1u, 9.0},
        {5, 2u, 12.0}
    });

    pmat out(8, 8, 2);
    palg::division_scaler(A, out, 3.0);

    EXPECT_DOUBLE_EQ(out.get(0, 0u), 2.0);
    EXPECT_DOUBLE_EQ(out.get(3, 1u), 3.0);
    EXPECT_DOUBLE_EQ(out.get(5, 2u), 4.0);
}

TEST(ParallelScalars, DivisionByZeroThrows)
{
    auto A = build_locked(4, 4, 2, {{0, 0u, 1.0}});
    pmat out(4, 4, 2);
    EXPECT_THROW(palg::division_scaler(A, out, 0.0), std::domain_error);
}

TEST(ParallelScalars, MultiplicationByOneIsIdentity)
{
    std::vector<entry> entries = {
        {0, 1u, 3.5}, {2, 0u, -1.0}, {5, 3u, 7.0}
    };
    auto A = build_locked(8, 4, 2, entries);
    pmat out(8, 4, 2);
    palg::multiplication_scaler(A, out, 1.0);

    for (auto &[r, c, v] : entries)
        EXPECT_DOUBLE_EQ(out.get(r, c), v) << "(" << r << "," << c << ")";
}

TEST(ParallelScalars, MatchesAcrossThreadCounts)
{
    std::vector<entry> entries;
    for (std::size_t r = 0; r < 8; ++r)
        entries.emplace_back(r, static_cast<uint32_t>(r % 4), static_cast<double>(r + 1));

    auto A1 = build_locked(8, 4, 1, entries);
    auto A2 = build_locked(8, 4, 2, entries);

    pmat out1(8, 4, 1), out2(8, 4, 2);
    palg::multiplication_scaler(A1, out1, 2.5);
    palg::multiplication_scaler(A2, out2, 2.5);

    for (auto &[r, c, v] : entries)
        EXPECT_DOUBLE_EQ(out1.get(r, c), out2.get(r, c))
            << "mismatch at (" << r << "," << c << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// matrix addition
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelMatrixAddition, SameSparsityPattern)
{
    auto A = build_locked(8, 8, 2, {
        {0, 0u, 1.0}, {0, 2u, 2.0},
        {4, 1u, 3.0}, {7, 7u, 4.0}
    });
    auto B = build_locked(8, 8, 2, {
        {0, 0u, 10.0}, {0, 2u, 20.0},
        {4, 1u, 30.0}, {7, 7u, 40.0}
    });

    auto C = palg::MatrixAddition(A, B);

    EXPECT_DOUBLE_EQ(C.get(0, 0u), 11.0);
    EXPECT_DOUBLE_EQ(C.get(0, 2u), 22.0);
    EXPECT_DOUBLE_EQ(C.get(4, 1u), 33.0);
    EXPECT_DOUBLE_EQ(C.get(7, 7u), 44.0);
}

TEST(ParallelMatrixAddition, DisjointSparsityPattern)
{
    auto A = build_locked(8, 8, 2, {{0, 0u, 1.0}, {4, 1u, 2.0}});
    auto B = build_locked(8, 8, 2, {{0, 1u, 3.0}, {4, 0u, 4.0}});

    auto C = palg::MatrixAddition(A, B);

    EXPECT_DOUBLE_EQ(C.get(0, 0u), 1.0);
    EXPECT_DOUBLE_EQ(C.get(0, 1u), 3.0);
    EXPECT_DOUBLE_EQ(C.get(4, 0u), 4.0);
    EXPECT_DOUBLE_EQ(C.get(4, 1u), 2.0);
    EXPECT_EQ(C.nnz(), 4u);
}

TEST(ParallelMatrixAddition, CancellationDropsZeros)
{
    auto A = build_locked(4, 4, 2, {{0, 0u, 5.0}, {1, 1u, 3.0}});
    auto B = build_locked(4, 4, 2, {{0, 0u, -5.0}, {1, 1u, 1.0}});

    auto C = palg::MatrixAddition(A, B);

    EXPECT_DOUBLE_EQ(C.get(0, 0u), 0.0);  // cancelled — dropped from CSR
    EXPECT_DOUBLE_EQ(C.get(1, 1u), 4.0);
    EXPECT_FALSE(C.contains(0, 0u));  // zero was filtered
}

TEST(ParallelMatrixAddition, ShapeMismatchThrows)
{
    auto A = build_locked(4, 4, 2, {});
    auto B = build_locked(4, 6, 2, {});
    EXPECT_THROW(palg::MatrixAddition(A, B), std::invalid_argument);
}

TEST(ParallelMatrixAddition, MatchesAcrossThreadCounts)
{
    std::vector<entry> a_entries = {{0, 0u, 1.0}, {2, 1u, 2.0}, {4, 2u, 3.0}, {6, 3u, 4.0}};
    std::vector<entry> b_entries = {{0, 1u, 5.0}, {2, 0u, 6.0}, {4, 3u, 7.0}, {6, 2u, 8.0}};

    auto A1 = build_locked(8, 4, 1, a_entries);
    auto B1 = build_locked(8, 4, 1, b_entries);
    auto A2 = build_locked(8, 4, 2, a_entries);
    auto B2 = build_locked(8, 4, 2, b_entries);

    auto C1 = palg::MatrixAddition(A1, B1);
    auto C2 = palg::MatrixAddition(A2, B2);

    for (std::size_t r = 0; r < 8; ++r)
        for (uint32_t c = 0; c < 4; ++c)
            EXPECT_DOUBLE_EQ(C1.get(r, c), C2.get(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// transpose
// ─────────────────────────────────────────────────────────────────────────────

TEST(ParallelTranspose, SquareDiagonal)
{
    std::vector<entry> entries;
    for (std::size_t i = 0; i < 8; ++i)
        entries.emplace_back(i, static_cast<uint32_t>(i), static_cast<double>(i + 1));

    auto A = build_locked(8, 8, 2, entries);
    auto T = palg::transpose(A);

    EXPECT_EQ(T.n_rows(), 8u);
    EXPECT_EQ(T.n_cols(), 8u);
    for (std::size_t i = 0; i < 8; ++i)
        EXPECT_DOUBLE_EQ(T.get(i, static_cast<uint32_t>(i)), static_cast<double>(i + 1));
}

TEST(ParallelTranspose, NonSquare)
{
    // 4 rows × 6 cols
    auto A = build_locked(4, 6, 2, {
        {0, 5u, 1.0},
        {1, 2u, 2.0},
        {3, 0u, 3.0}
    });

    auto T = palg::transpose(A);

    EXPECT_EQ(T.n_rows(), 6u);
    EXPECT_EQ(T.n_cols(), 4u);

    EXPECT_DOUBLE_EQ(T.get(5, 0u), 1.0);  // A(0,5) → T(5,0)
    EXPECT_DOUBLE_EQ(T.get(2, 1u), 2.0);  // A(1,2) → T(2,1)
    EXPECT_DOUBLE_EQ(T.get(0, 3u), 3.0);  // A(3,0) → T(0,3)
    EXPECT_EQ(T.nnz(), 3u);
}

TEST(ParallelTranspose, DoubleTransposeIsIdentity)
{
    std::vector<entry> entries = {
        {0, 1u, 1.5}, {0, 3u, 2.5},
        {2, 0u, 3.5}, {3, 2u, 4.5}
    };
    auto A  = build_locked(4, 4, 2, entries);
    auto T  = palg::transpose(A);
    auto TT = palg::transpose(T);

    for (auto &[r, c, v] : entries)
        EXPECT_DOUBLE_EQ(TT.get(r, c), v) << "(" << r << "," << c << ")";
    EXPECT_EQ(TT.nnz(), A.nnz());
}

TEST(ParallelTranspose, MatchesAcrossThreadCounts)
{
    std::vector<entry> entries = {
        {0, 3u, 1.0}, {1, 1u, 2.0}, {2, 2u, 3.0},
        {4, 0u, 4.0}, {5, 3u, 5.0}, {7, 1u, 6.0}
    };
    auto A1 = build_locked(8, 4, 1, entries);
    auto A2 = build_locked(8, 4, 2, entries);

    auto T1 = palg::transpose(A1);
    auto T2 = palg::transpose(A2);

    EXPECT_EQ(T1.n_rows(), T2.n_rows());
    for (std::size_t r = 0; r < T1.n_rows(); ++r)
        for (uint32_t c = 0; c < 8; ++c)
            EXPECT_DOUBLE_EQ(T1.get(r, c), T2.get(r, c))
                << "mismatch at (" << r << "," << c << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// spgemm
// ─────────────────────────────────────────────────────────────────────────────

// Dense matrix multiply reference
static std::vector<std::vector<double>>
dense_matmul(const std::vector<std::vector<double>> &A,
             const std::vector<std::vector<double>> &B,
             std::size_t m, std::size_t k, std::size_t n)
{
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t l = 0; l < k; ++l)
            if (A[i][l] != 0.0)
                for (std::size_t j = 0; j < n; ++j)
                    C[i][j] += A[i][l] * B[l][j];
    return C;
}

// Extract parallel_matrix into dense 2D array
static std::vector<std::vector<double>>
to_dense(pmat &m)
{
    std::vector<std::vector<double>> d(m.n_rows(), std::vector<double>(m.n_cols(), 0.0));
    m.for_each_nnz_row([&d](const auto &row, auto global_r) {
        row.for_each_element([&d, global_r](uint32_t c, double v) {
            d[static_cast<std::size_t>(global_r)][static_cast<std::size_t>(c)] = v;
        });
    });
    return d;
}

TEST(ParallelSpgemm, IdentityTimesMatrix)
{
    // I * A = A
    std::size_t N = 4;
    std::vector<entry> I_entries, A_entries;
    for (std::size_t i = 0; i < N; ++i)
        I_entries.emplace_back(i, static_cast<uint32_t>(i), 1.0);
    A_entries = {{0, 1u, 2.0}, {1, 0u, 3.0}, {2, 2u, 4.0}, {3, 1u, 5.0}};

    auto I_mat = build_locked(N, N, 2, I_entries);
    auto A_mat = build_locked(N, N, 2, A_entries);

    auto C = palg::spgemm(I_mat, A_mat);

    for (auto &[r, c, v] : A_entries)
        EXPECT_DOUBLE_EQ(C.get(r, c), v) << "(" << r << "," << c << ")";
    EXPECT_EQ(C.nnz(), A_entries.size());
}

TEST(ParallelSpgemm, DiagonalTimesMatrix)
{
    // D * A where D = diag(2,3,4,5)
    std::vector<entry> D_entries = {{0,0u,2.0},{1,1u,3.0},{2,2u,4.0},{3,3u,5.0}};
    std::vector<entry> A_entries = {{0,1u,1.0},{1,2u,1.0},{2,0u,1.0},{3,3u,1.0}};

    auto D = build_locked(4, 4, 2, D_entries);
    auto A = build_locked(4, 4, 2, A_entries);
    auto C = palg::spgemm(D, A);

    EXPECT_DOUBLE_EQ(C.get(0, 1u), 2.0);
    EXPECT_DOUBLE_EQ(C.get(1, 2u), 3.0);
    EXPECT_DOUBLE_EQ(C.get(2, 0u), 4.0);
    EXPECT_DOUBLE_EQ(C.get(3, 3u), 5.0);
}

TEST(ParallelSpgemm, RectangularMatrices)
{
    // A: 4×3, B: 3×5 → C: 4×5
    std::vector<entry> A_entries = {
        {0, 0u, 1.0}, {0, 2u, 2.0},
        {1, 1u, 3.0},
        {2, 0u, 4.0}, {2, 1u, 5.0},
        {3, 2u, 6.0}
    };
    std::vector<entry> B_entries = {
        {0, 0u, 1.0}, {0, 3u, 2.0},
        {1, 1u, 3.0}, {1, 4u, 4.0},
        {2, 2u, 5.0}
    };

    auto A = build_locked(4, 3, 2, A_entries);
    auto B = build_locked(3, 5, 2, B_entries);
    auto C = palg::spgemm(A, B);

    EXPECT_EQ(C.n_rows(), 4u);
    EXPECT_EQ(C.n_cols(), 5u);

    // Build dense reference
    std::vector<std::vector<double>> dA(4, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> dB(3, std::vector<double>(5, 0.0));
    for (auto &[r, c, v] : A_entries) dA[r][c] = v;
    for (auto &[r, c, v] : B_entries) dB[r][c] = v;
    auto dC = dense_matmul(dA, dB, 4, 3, 5);

    for (std::size_t r = 0; r < 4; ++r)
        for (uint32_t c = 0; c < 5; ++c)
            EXPECT_DOUBLE_EQ(C.get(r, c), dC[r][c]) << "(" << r << "," << c << ")";
}

TEST(ParallelSpgemm, DimensionMismatchThrows)
{
    auto A = build_locked(4, 3, 2, {});
    auto B = build_locked(4, 4, 2, {});  // B.n_rows()=4 != A.n_cols()=3
    EXPECT_THROW(palg::spgemm(A, B), std::invalid_argument);
}

TEST(ParallelSpgemm, MatchesAcrossThreadCounts)
{
    std::vector<entry> A_entries = {
        {0, 0u, 1.0}, {1, 1u, 2.0}, {2, 0u, 3.0}, {3, 1u, 4.0},
        {4, 0u, 1.0}, {5, 1u, 2.0}, {6, 0u, 3.0}, {7, 1u, 4.0}
    };
    std::vector<entry> B_entries = {
        {0, 0u, 5.0}, {0, 1u, 6.0},
        {1, 0u, 7.0}, {1, 1u, 8.0}
    };

    auto A1 = build_locked(8, 2, 1, A_entries);
    auto B1 = build_locked(2, 2, 1, B_entries);
    auto A2 = build_locked(8, 2, 2, A_entries);
    auto B2 = build_locked(2, 2, 2, B_entries);

    auto C1 = palg::spgemm(A1, B1);
    auto C2 = palg::spgemm(A2, B2);

    for (std::size_t r = 0; r < 8; ++r)
        for (uint32_t c = 0; c < 2; ++c)
            EXPECT_DOUBLE_EQ(C1.get(r, c), C2.get(r, c))
                << "1 vs 2 threads at (" << r << "," << c << ")";
}
