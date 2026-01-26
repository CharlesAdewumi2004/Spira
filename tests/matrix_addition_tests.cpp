#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <spira/matrix/matrix.hpp>
#include <spira/matrix/row.hpp>
#include <spira/algorithms/matrix_addition.hpp>

using LayoutTag = spira::layout::tags::soa_tag;

using Index = std::size_t;
using Value = double;

using Row = spira::row<LayoutTag, Index, Value>;
using Mat = spira::matrix<LayoutTag, Index, Value>;

namespace
{

    void insert_row(Row &r, std::initializer_list<std::pair<Index, Value>> xs)
    {
        for (auto [c, v] : xs)
            r.insert(c, v);
    }

    void insert_mat(Mat &m, std::initializer_list<std::tuple<Index, Index, Value>> xs)
    {
        for (auto [r, c, v] : xs)
            m.insert(r, c, v);
    }

}

// --------------------
// addRows tests
// --------------------

TEST(AddRows, DisjointColumns_MergesBoth)
{
    Mat dummy(1, 10);
    Row A(0, 10);
    Row B(0, 10);
    Row out(0, 10);

    insert_row(A, {{1, 1.0}, {4, 4.0}});
    insert_row(B, {{2, 2.0}, {8, 8.0}});

    spira::algorithms::addRows(A, B, out);   

    out.flush();

    auto it = out.begin();
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 1u);
    EXPECT_EQ(*it.val_ptr, 1.0);
    ++it;
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 2u);
    EXPECT_EQ(*it.val_ptr, 2.0);
    ++it;
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 4u);
    EXPECT_EQ(*it.val_ptr, 4.0);
    ++it;
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 8u);
    EXPECT_EQ(*it.val_ptr, 8.0);
    ++it;
    EXPECT_EQ(it, out.end());
}

TEST(AddRows, OverlappingColumns_SumsValues)
{
    Row A(0, 10);
    Row B(0, 10);
    Row out(0, 10);

    insert_row(A, {{1, 1.5}, {4, 4.0}});
    insert_row(B, {{1, 2.5}, {8, 8.0}});

    spira::algorithms::addRows(A, B, out);

    out.flush();

    auto it = out.begin();
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 1u);
    EXPECT_EQ(*it.val_ptr, 4.0);
    ++it;
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 4u);
    EXPECT_EQ(*it.val_ptr, 4.0);
    ++it;
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 8u);
    EXPECT_EQ(*it.val_ptr, 8.0);
    ++it;
    EXPECT_EQ(it, out.end());
}

TEST(AddRows, OverlappingColumns_ZeroSumIsDropped)
{
    Row A(0, 10);
    Row B(0, 10);
    Row out(0, 10);

    insert_row(A, {{3, 5.0}});
    insert_row(B, {{3, -5.0}});

    spira::algorithms::addRows(A, B, out);

    out.flush();

    EXPECT_EQ(out.begin(), out.end())
        << "If this fails, your zero-sum branch is inverted (you are inserting zeros).";
}

TEST(AddRows, OneSideEmpty_CopiesOther)
{
    Row A(0, 10);
    Row B(0, 10);
    Row out(0, 10);

    insert_row(B, {{0, 1.0}, {9, 2.0}});

    spira::algorithms::addRows(A, B, out);

    out.flush();

    auto it = out.begin();
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 0u);
    EXPECT_EQ(*it.val_ptr, 1.0);
    ++it;
    ASSERT_NE(it, out.end());
    EXPECT_EQ(*it.col_ptr, 9u);
    EXPECT_EQ(*it.val_ptr, 2.0);
    ++it;
    EXPECT_EQ(it, out.end());
}

// --------------------
// MatrixAddition tests
// --------------------

TEST(MatrixAddition, ThrowsOnShapeMismatch)
{
    Mat A(2, 3);
    Mat B(3, 2);

    EXPECT_THROW(spira::algorithms::MatrixAddition(A, B), std::invalid_argument);
}

TEST(MatrixAddition, AddsMatricesElementwise_Sparse)
{
    Mat A(3, 4);
    Mat B(3, 4);

    insert_mat(A, {
                      {0, 0, 1.0},
                      {0, 3, 2.0},
                      {2, 1, 3.0},
                  });

    insert_mat(B, {
                      {0, 0, 10.0},
                      {1, 2, 20.0},
                      {2, 1, -3.0},
                  });

    auto C = spira::algorithms::MatrixAddition(A, B);

    auto [r, c] = C.get_shape();
    EXPECT_EQ(r, 3u);
    EXPECT_EQ(c, 4u);

    EXPECT_EQ(C.get(0, 0), 11.0);

    EXPECT_EQ(C.get(0, 3), 2.0);

    EXPECT_EQ(C.get(1, 2), 20.0);

    EXPECT_EQ(C.get(2, 1), 0.0);

    EXPECT_EQ(C.get(2, 3), 0.0);

    EXPECT_EQ(A.get(0, 0), 1.0);
    EXPECT_EQ(B.get(0, 0), 10.0);
    EXPECT_EQ(A.get(2, 1), 3.0);
    EXPECT_EQ(B.get(2, 1), -3.0);
}

TEST(MatrixAddition, ZeroMatrix_AdditionKeepsOriginal)
{
    Mat A(2, 2);
    Mat Z(2, 2);

    insert_mat(A, {
                      {0, 1, 7.0},
                      {1, 0, 9.0},
                  });

    auto C = spira::algorithms::MatrixAddition(A, Z);

    EXPECT_EQ(C.get(0, 1), 7.0);
    EXPECT_EQ(C.get(1, 0), 9.0);
    EXPECT_EQ(C.get(0, 0), 0.0);
    EXPECT_EQ(C.get(1, 1), 0.0);
}
