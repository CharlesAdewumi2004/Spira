// transpose_tests.cpp
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>
#include <tuple>

#include <spira/spira.hpp>
#include <spira/algorithms/transpose.hpp> // adjust to your actual header

namespace
{

    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    using Mat = spira::matrix<Layout, I, V>;

    // Build from (r,c,val) triplets
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    Mat<Layout, I, V> make_mat(I rows, I cols, std::initializer_list<std::tuple<I, I, V>> entries)
    {
        Mat<Layout, I, V> M(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
        for (auto [r, c, v] : entries)
        {
            M.insert(r, c, v);
        }
        return M;
    }

    template <class Layout, class I, class V>
    void expect_dense_equal(const Mat<Layout, I, V> &A, const std::vector<std::vector<V>> &ref)
    {
        ASSERT_EQ(static_cast<I>(A.n_rows()), static_cast<I>(ref.size()));
        ASSERT_EQ(static_cast<I>(A.n_cols()), static_cast<I>(ref.empty() ? 0 : ref[0].size()));

        for (I i = 0; i < static_cast<I>(A.n_rows()); ++i)
        {
            for (I j = 0; j < static_cast<I>(A.n_cols()); ++j)
            {
                if constexpr (std::is_floating_point_v<V>)
                {
                    EXPECT_NEAR(A.get(i, j), ref[i][j], static_cast<V>(1e-9)) << "at (" << i << "," << j << ")";
                }
                else
                {
                    EXPECT_EQ(A.get(i, j), ref[i][j]) << "at (" << i << "," << j << ")";
                }
            }
        }
    }

}

using LayoutTag = spira::layout::tags::soa_tag;
using I = size_t;

// 1) Shape flips: (m x n) -> (n x m)
TEST(Transpose, ShapeFlips)
{
    using V = int;
    Mat<LayoutTag, I, V> A(3, 5);

    auto AT = spira::algorithms::transpose(A);

    EXPECT_EQ(AT.n_rows(), 5u);
    EXPECT_EQ(AT.n_cols(), 3u);
}

// 2) Empty stays empty (all gets are 0)
TEST(Transpose, EmptyMatrix)
{
    using V = int;
    Mat<LayoutTag, I, V> A(2, 4);

    auto AT = spira::algorithms::transpose(A);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_EQ(AT.get(i, j), 0);
}

// 3) Single element moves to swapped coordinate
TEST(Transpose, SingleElement)
{
    using V = int;
    auto A = make_mat<LayoutTag, I, V>(4, 3, {{2, 1, 9}});

    auto AT = spira::algorithms::transpose(A);

    EXPECT_EQ(AT.n_rows(), 3u);
    EXPECT_EQ(AT.n_cols(), 4u);

    EXPECT_EQ(AT.get(1, 2), 9);
    EXPECT_EQ(AT.get(2, 1), 0);
}

// 4) General sparse example
TEST(Transpose, GeneralSparseExample)
{
    using V = int;

    // A (3x4):
    // [0 5 0 0]
    // [7 0 0 2]
    // [0 0 3 0]
    auto A = make_mat<LayoutTag, I, V>(3, 4, {{0, 1, 5}, {1, 0, 7}, {1, 3, 2}, {2, 2, 3}});

    auto AT = spira::algorithms::transpose(A);

    // AT (4x3):
    // [0 7 0]
    // [5 0 0]
    // [0 0 3]
    // [0 2 0]
    std::vector<std::vector<V>> ref = {
        {0, 7, 0},
        {5, 0, 0},
        {0, 0, 3},
        {0, 2, 0}};

    expect_dense_equal(AT, ref);
}

// 5) Involution: transpose(transpose(A)) == A
TEST(Transpose, DoubleTransposeIsOriginal)
{
    using V = int;

    auto A = make_mat<LayoutTag, I, V>(5, 3, {{0, 0, 1}, {0, 2, 4}, {2, 1, -3}, {4, 2, 8}});

    auto AT = spira::algorithms::transpose(A);
    auto ATT = spira::algorithms::transpose(AT);

    // Compare dense via get()
    ASSERT_EQ(ATT.n_rows(), A.n_rows());
    ASSERT_EQ(ATT.n_cols(), A.n_cols());

    for (int i = 0; i < static_cast<int>(A.n_rows()); ++i)
        for (int j = 0; j < static_cast<int>(A.n_cols()); ++j)
            EXPECT_EQ(ATT.get(i, j), A.get(i, j)) << "at (" << i << "," << j << ")";
}

// 6) Floating point values (NEAR)
TEST(Transpose, DoubleValues)
{
    using V = double;

    auto A = make_mat<LayoutTag, I, V>(2, 3, {{0, 2, 0.25}, {1, 0, -1.5}});

    auto AT = spira::algorithms::transpose(A);

    EXPECT_NEAR(AT.get(2, 0), 0.25, 1e-9);
    EXPECT_NEAR(AT.get(0, 1), -1.5, 1e-9);
}

// 7) Dense-ish random pattern sanity (still small)
TEST(Transpose, ManyEntriesSanity)
{
    using V = int;
    Mat<LayoutTag, I, V> A(3, 3);

    A.insert(0, 0, 1);
    A.insert(1, 1, 2);
    A.insert(2, 2, 3);
    A.insert(0, 2, 4);
    A.insert(2, 0, 5);

    auto AT = spira::algorithms::transpose(A);

    EXPECT_EQ(AT.get(0, 0), 1);
    EXPECT_EQ(AT.get(1, 1), 2);
    EXPECT_EQ(AT.get(2, 2), 3);
    EXPECT_EQ(AT.get(2, 0), 4);
    EXPECT_EQ(AT.get(0, 2), 5);
}
