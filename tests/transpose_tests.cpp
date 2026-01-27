// test_transpose.cpp
#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

#include <spira/matrix/matrix.hpp>
#include <spira/algorithms/transpose.hpp>

using Layout = spira::layout::tags::aos_tag;

using Index = std::size_t;
using Value = double;

using Matrix = spira::matrix<Layout, Index, Value>;

namespace
{

    void insert_all(Matrix &m, std::initializer_list<std::tuple<Index, Index, Value>> entries)
    {
        for (auto [r, c, v] : entries)
        {
            m.insert(r, c, v);
        }
    }

    void expect_same_shape(const Matrix &a, const Matrix &b)
    {
        auto [ar, ac] = a.shape();
        auto [br, bc] = b.shape();
        EXPECT_EQ(ar, br);
        EXPECT_EQ(ac, bc);
    }

}

// ----------------------------
// transpose() tests
// ----------------------------

TEST(Transpose, EmptyMatrix_ShapeSwapsAndStaysEmpty)
{
    Matrix m(3, 5);
    auto out = spira::algorithms::transpose(m);

    auto [r, c] = out.shape();
    EXPECT_EQ(r, 5u);
    EXPECT_EQ(c, 3u);

    EXPECT_EQ(out.get(0, 0), 0.0);
    EXPECT_EQ(out.get(4, 2), 0.0);

    // Input should be unchanged
    auto [mr, mc] = m.shape();
    EXPECT_EQ(mr, 3u);
    EXPECT_EQ(mc, 5u);
}

TEST(Transpose, SingleElement_GoesToCorrectPlace)
{
    Matrix m(4, 6);
    m.insert(2, 5, 7.5);

    auto out = spira::algorithms::transpose(m);

    EXPECT_EQ(out.get(5, 2), 7.5);

    EXPECT_EQ(m.get(2, 5), 7.5);

    EXPECT_EQ(out.get(2, 3), 0.0);
}

TEST(Transpose, MultipleEntries_AllMoveCorrectly)
{
    Matrix m(3, 4);
    insert_all(m, {
                      {0, 0, 1.0},
                      {0, 3, 2.0},
                      {2, 1, 3.0},
                  });

    auto out = spira::algorithms::transpose(m);

    EXPECT_EQ(out.get(0, 0), 1.0);
    EXPECT_EQ(out.get(3, 0), 2.0);
    EXPECT_EQ(out.get(1, 2), 3.0);

    EXPECT_EQ(out.get(2, 0), 0.0);
    EXPECT_EQ(out.get(0, 2), 0.0);

    EXPECT_EQ(m.get(0, 3), 2.0);
    EXPECT_EQ(m.get(2, 1), 3.0);
}

TEST(Transpose, DoubleTranspose_ReturnsOriginalValuesAndShape)
{
    Matrix m(2, 3);
    insert_all(m, {
                      {0, 1, 11.0},
                      {1, 2, 22.0},
                  });

    auto t1 = spira::algorithms::transpose(m);
    auto t2 = spira::algorithms::transpose(t1);

    expect_same_shape(m, t2);

    EXPECT_EQ(t2.get(0, 1), 11.0);
    EXPECT_EQ(t2.get(1, 2), 22.0);

    EXPECT_EQ(t2.get(0, 2), 0.0);
}

// ----------------------------
// transpose_itself() tests
// ----------------------------

TEST(TransposeItself, ThrowsOnNonSquare)
{
    Matrix m(2, 3);
    EXPECT_THROW(spira::algorithms::transpose_itself(m), std::logic_error);
}

TEST(TransposeItself, WorksOnSquare_SingleElement)
{
    Matrix m(4, 4);
    m.insert(1, 3, 9.0);

    spira::algorithms::transpose_itself(m);

    EXPECT_EQ(m.get(3, 1), 9.0);
    EXPECT_EQ(m.get(1, 3), 0.0);
}

TEST(TransposeItself, WorksOnSquare_ManyEntries)
{
    Matrix m(3, 3);
    insert_all(m, {
                      {0, 2, 5.0},
                      {1, 0, 6.0},
                      {2, 1, 7.0},
                  });

    spira::algorithms::transpose_itself(m);

    EXPECT_EQ(m.get(2, 0), 5.0);
    EXPECT_EQ(m.get(0, 1), 6.0);
    EXPECT_EQ(m.get(1, 2), 7.0);

    EXPECT_EQ(m.get(0, 2), 0.0);
    EXPECT_EQ(m.get(1, 0), 0.0);
    EXPECT_EQ(m.get(2, 1), 0.0);
}

TEST(TransposeItself, DoubleTransposeItself_ReturnsOriginal)
{
    Matrix m(3, 3);
    insert_all(m, {
                      {0, 1, 1.25},
                      {2, 2, 4.5},
                      {1, 0, -3.0},
                  });

    const auto a01 = m.get(0, 1);
    const auto a22 = m.get(2, 2);
    const auto a10 = m.get(1, 0);

    spira::algorithms::transpose_itself(m);
    spira::algorithms::transpose_itself(m);

    EXPECT_EQ(m.get(0, 1), a01);
    EXPECT_EQ(m.get(2, 2), a22);
    EXPECT_EQ(m.get(1, 0), a10);
}
