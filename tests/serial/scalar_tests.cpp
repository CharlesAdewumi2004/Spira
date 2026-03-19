// test_scalars.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>

#include <spira/matrix/matrix.hpp>
#include <spira/serial/scalars.hpp>

using LayoutTag = spira::layout::tags::aos_tag;

using Index = std::size_t;
using Value = double;

using Mat = spira::matrix<LayoutTag, Index, Value>;

namespace {

void insert(Mat& m, std::initializer_list<std::tuple<Index, Index, Value>> xs) {
    for (auto [r, c, v] : xs) m.insert(r, c, v);
}

void expect_near(Value a, Value b, Value eps = 1e-12) {
    EXPECT_NEAR(a, b, eps);
}

}

TEST(Scalers, MultiplicationScaler_ScalesExistingEntries)
{
    Mat m(3, 4);
    insert(m, {
        {0, 0, 1.5},
        {0, 3, -2.0},
        {2, 1, 4.0},
    });

    spira::serial::algorithms::multiplication_scaler(m, 3.0);

    expect_near(m.get(0, 0),  4.5);
    expect_near(m.get(0, 3), -6.0);
    expect_near(m.get(2, 1), 12.0);

    // missing entries remain zero
    expect_near(m.get(1, 1), 0.0);
    expect_near(m.get(2, 3), 0.0);
}

TEST(Scalers, DivisionScaler_ScalesExistingEntries)
{
    Mat m(2, 3);
    insert(m, {
        {0, 2, 10.0},
        {1, 0, -6.0},
    });

    spira::serial::algorithms::division_scaler(m, 2.0);

    expect_near(m.get(0, 2),  5.0);
    expect_near(m.get(1, 0), -3.0);

    // missing entry remains zero
    expect_near(m.get(0, 0), 0.0);
}

TEST(Scalers, MultiplyThenDivide_ReturnsOriginal_ForNonZeroScaler)
{
    Mat m(3, 3);
    insert(m, {
        {0, 1,  1.25},
        {2, 2, -9.0},
        {1, 0,  0.5},
    });

    // snapshot original values
    const Value a01 = m.get(0, 1);
    const Value a22 = m.get(2, 2);
    const Value a10 = m.get(1, 0);

    const Value s = 7.0;

    spira::serial::algorithms::multiplication_scaler(m, s);
    spira::serial::algorithms::division_scaler(m, s);

    expect_near(m.get(0, 1), a01);
    expect_near(m.get(2, 2), a22);
    expect_near(m.get(1, 0), a10);

    // still sparse: missing entries remain zero
    expect_near(m.get(0, 0), 0.0);
    expect_near(m.get(2, 0), 0.0);
}

TEST(Scalers, MultiplyByZero_MakesEntriesZero)
{
    Mat m(2, 2);
    insert(m, {
        {0, 0, 3.0},
        {1, 1, -4.0},
    });

    spira::serial::algorithms::multiplication_scaler(m, 0.0);

    expect_near(m.get(0, 0), 0.0);
    expect_near(m.get(1, 1), 0.0);
}

TEST(Scalers, DivisionByZero_Behavior)
{
    Mat m(1, 3);
    insert(m, {
        {0, 1, 2.0},
    });
    // Zero divisor check fires before open assert — matrix is open by default
    EXPECT_THROW(spira::serial::algorithms::division_scaler(m, 0.0), std::domain_error);
}
