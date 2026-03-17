// spgemm_tests.cpp
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <type_traits>

#include <spira/spira.hpp>
#include <spira/serial/spgemm.hpp>

namespace
{

    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    using Mat = spira::matrix<Layout, I, V>;

    // Helper: build a locked sparse matrix from triplets (i,j,val)
    template <class Layout, class I, class V>
    Mat<Layout, I, V> make_mat(I rows, I cols, std::initializer_list<std::tuple<I, I, V>> entries)
    {
        Mat<Layout, I, V> M(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        for (auto [r, c, v] : entries)
        {
            M.insert(r, c, v);
        }
        M.lock();
        return M;
    }

    // Helper: dense reference multiply (for small sizes in tests)
    template <class Layout, class I, class V>
    std::vector<std::vector<V>> dense_ref_mul(const Mat<Layout, I, V> &A, const Mat<Layout, I, V> &B)
    {
        const I m = static_cast<I>(A.n_rows());
        const I n = static_cast<I>(A.n_cols());
        const I p = static_cast<I>(B.n_cols());

        std::vector<std::vector<V>> C(m, std::vector<V>(p, V{}));

        for (I i = 0; i < m; ++i)
        {
            for (I k = 0; k < n; ++k)
            {
                const V a = A.get(i, k);
                if (a == V{})
                    continue;
                for (I j = 0; j < p; ++j)
                {
                    const V b = B.get(k, j);
                    if (b == V{})
                        continue;
                    C[i][j] += a * b;
                }
            }
        }
        return C;
    }

    // Helper: compare sparse result to dense reference
    template <class Layout, class I, class V>
    void expect_equal_to_dense(const Mat<Layout, I, V> &C, const std::vector<std::vector<V>> &ref)
    {
        ASSERT_EQ(static_cast<I>(C.n_rows()), static_cast<I>(ref.size()));
        ASSERT_EQ(static_cast<I>(C.n_cols()), static_cast<I>(ref.empty() ? 0 : ref[0].size()));

        for (I i = 0; i < static_cast<I>(C.n_rows()); ++i)
        {
            for (I j = 0; j < static_cast<I>(C.n_cols()); ++j)
            {
                if constexpr (std::is_floating_point_v<V>)
                {
                    EXPECT_NEAR(C.get(i, j), ref[i][j], static_cast<V>(1e-9)) << "at (" << i << "," << j << ")";
                }
                else
                {
                    EXPECT_EQ(C.get(i, j), ref[i][j]) << "at (" << i << "," << j << ")";
                }
            }
        }
    }

}

// -------------------------
// Core correctness tests
// -------------------------

using LayoutTag = spira::layout::tags::aos_tag;
using I = size_t;

// 1) Dimension mismatch throws (shape check fires first in spgemm, no lock needed)
TEST(SpGEMM, ThrowsOnIncompatibleShapes)
{
    using V = double;
    Mat<LayoutTag, I, V> A(2ull, 3);
    Mat<LayoutTag, I, V> B(2ull, 2);

    EXPECT_THROW((spira::algorithms::spgemm(A, B)), std::invalid_argument);
}

// 2) Multiply by zero matrix gives zero matrix
TEST(SpGEMM, ZeroMatrixGivesZero)
{
    using V = int;
    auto A = make_mat<LayoutTag, I, V>(3, 4, {{0, 1, 2}, {2, 3, 5}});

    Mat<LayoutTag, I, V> Z(4, 2); // all zeros
    Z.lock();
    auto C = spira::algorithms::spgemm(A, Z);

    auto ref = dense_ref_mul(A, Z);
    expect_equal_to_dense(C, ref);
}

// 3) Identity: A * I = A
TEST(SpGEMM, RightIdentity)
{
    using V = int;
    auto A = make_mat<LayoutTag, I, V>(3, 3, {{0, 0, 1}, {0, 2, 4}, {1, 1, 3}, {2, 0, -2}, {2, 2, 7}});

    auto Iden = make_mat<LayoutTag, I, V>(3, 3, {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}});

    auto C = spira::algorithms::spgemm(A, Iden);

    auto ref = dense_ref_mul(A, Iden);
    expect_equal_to_dense(C, ref);
}

// 4) Small hand-checkable example
TEST(SpGEMM, SmallExampleMatchesReference)
{
    using V = int;

    // A (2x3):  [1 0 2] / [0 3 0]
    auto A = make_mat<LayoutTag, I, V>(2, 3, {{0, 0, 1}, {0, 2, 2}, {1, 1, 3}});

    // B (3x2):  [0 4] / [5 0] / [0 6]
    auto B = make_mat<LayoutTag, I, V>(3, 2, {{0, 1, 4}, {1, 0, 5}, {2, 1, 6}});

    auto C = spira::algorithms::spgemm(A, B);

    EXPECT_EQ(C.n_rows(), 2u);
    EXPECT_EQ(C.n_cols(), 2u);
    EXPECT_EQ(C.get(0, 0), 0);
    EXPECT_EQ(C.get(0, 1), 16);
    EXPECT_EQ(C.get(1, 0), 15);
    EXPECT_EQ(C.get(1, 1), 0);

    auto ref = dense_ref_mul(A, B);
    expect_equal_to_dense(C, ref);
}

// 5) Rectangular shapes work
TEST(SpGEMM, RectangularWorks)
{
    using V = int;
    auto A = make_mat<LayoutTag, I, V>(4, 3, {{0, 0, 1}, {0, 2, 2}, {2, 1, 3}, {3, 0, 4}});

    auto B = make_mat<LayoutTag, I, V>(3, 5, {{0, 1, 7}, {0, 4, 1}, {1, 2, 2}, {2, 1, 3}, {2, 3, 5}});

    auto C = spira::algorithms::spgemm(A, B);
    auto ref = dense_ref_mul(A, B);
    expect_equal_to_dense(C, ref);
}

// 6) Non-commutative sanity: generally A*B != B*A
TEST(SpGEMM, NonCommutativeSanity)
{
    using V = int;
    auto A = make_mat<LayoutTag, I, V>(2, 2, {{0, 0, 1}, {1, 0, 2}});
    auto B = make_mat<LayoutTag, I, V>(2, 2, {{0, 1, 3}, {1, 1, 4}});

    auto AB = spira::algorithms::spgemm(A, B);
    auto BA = spira::algorithms::spgemm(B, A);

    bool any_diff = false;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            any_diff |= (AB.get(i, j) != BA.get(i, j));

    EXPECT_TRUE(any_diff);
}

// -------------------------
// "gotcha" tests for sparse implementation details
// -------------------------

// 7) Accumulation test: multiple k contribute to same (i,j)
TEST(SpGEMM, AccumulatesContributionsCorrectly)
{
    using V = int;

    auto A = make_mat<LayoutTag, I, V>(1, 3, {{0, 0, 1}, {0, 1, 1}, {0, 2, 1}});

    auto B = make_mat<LayoutTag, I, V>(3, 1, {{0, 0, 2}, {1, 0, 3}, {2, 0, 4}});

    auto C = spira::algorithms::spgemm(A, B);
    EXPECT_EQ(C.get(0, 0), 9);

    auto ref = dense_ref_mul(A, B);
    expect_equal_to_dense(C, ref);
}

// 8) Floating point version (NEAR compare)
TEST(SpGEMM, DoubleWorks)
{
    using V = double;

    auto A = make_mat<LayoutTag, I, V>(2, 2, {{0, 0, 0.5}, {1, 1, 2.0}});
    auto B = make_mat<LayoutTag, I, V>(2, 2, {{0, 1, 4.0}, {1, 0, -1.5}});

    auto C = spira::algorithms::spgemm(A, B);
    auto ref = dense_ref_mul(A, B);
    expect_equal_to_dense(C, ref);
}
