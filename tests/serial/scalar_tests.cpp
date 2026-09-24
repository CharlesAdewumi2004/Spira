// tests/serial/scalar_tests.cpp
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

TEST(Scalers, InPlaceScaleAfterEarlierLockIsCommitted)
{
    // Regression: the in-place path used to write straight into the rows
    // without marking them dirty, so the merge on the next lock() skipped them
    // and the scaled values were lost.
    Mat m(2, 2);
    insert(m, {
        {0, 0, 2.0},
        {1, 1, 3.0},
    });
    m.lock();
    m.open();

    spira::serial::algorithms::multiplication_scaler(m, 10.0);
    m.lock();
    expect_near(m.get(0, 0), 20.0);
    expect_near(m.get(1, 1), 30.0);

    m.open();
    spira::serial::algorithms::division_scaler(m, 4.0);
    m.lock();
    expect_near(m.get(0, 0), 5.0);
    expect_near(m.get(1, 1), 7.5);
}

TEST(Scalers, InPlaceScaleWithPendingEditsScalesThemToo)
{
    Mat m(1, 3);
    insert(m, {{0, 0, 1.0}});
    m.lock();
    m.open();
    insert(m, {{0, 2, 4.0}}); // still in the buffer when the scaler runs

    spira::serial::algorithms::multiplication_scaler(m, 2.0);
    EXPECT_TRUE(m.is_open());
    m.lock();
    expect_near(m.get(0, 0), 2.0);
    expect_near(m.get(0, 2), 8.0);
}

TEST(Scalers, InPlaceScaleOnWideRowBeforeFirstLock)
{
    // More entries in one row than the buffer's initial reserve (64): the old
    // path appended to the buffer while iterating it.
    constexpr Index n = 200;
    Mat m(1, n);
    for (Index c = 0; c < n; ++c)
        m.insert(0, c, static_cast<Value>(c + 1));

    spira::serial::algorithms::multiplication_scaler(m, 3.0);
    m.lock();
    ASSERT_EQ(m.nnz(), n);
    for (Index c = 0; c < n; ++c)
        expect_near(m.get(0, c), 3.0 * static_cast<Value>(c + 1));
}
