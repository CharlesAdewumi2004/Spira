#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <spira/spira.hpp>

namespace {

using namespace spira;
using namespace spira::algorithms;

// ─────────────────────────────────────────────────────
// Aliases
// ─────────────────────────────────────────────────────

using AosMat = spira::matrix<layout::tags::aos_tag, uint32_t, double>;
using SoaMat = spira::matrix<layout::tags::soa_tag, uint32_t, double>;
using SoaMatF = spira::matrix<layout::tags::soa_tag, uint32_t, float>;
using AosMatI = spira::matrix<layout::tags::aos_tag, uint32_t, int>;

// ─────────────────────────────────────────────────────
// Single-row accumulate
// ─────────────────────────────────────────────────────

TEST(AccumulateSingleRow, EmptyRowReturnsZero) {
    AosMat mat(3, 5);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 0.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 1), 0.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 2), 0.0);
}

TEST(AccumulateSingleRow, SingleElement) {
    AosMat mat(3, 5);
    mat.insert(1, 2, 42.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 1), 42.0);
}

TEST(AccumulateSingleRow, MultipleElementsSameRow) {
    AosMat mat(2, 10);
    mat.insert(0, 0, 1.0);
    mat.insert(0, 3, 2.0);
    mat.insert(0, 7, 3.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 6.0);
}

TEST(AccumulateSingleRow, DoesNotIncludeOtherRows) {
    AosMat mat(3, 5);
    mat.insert(0, 0, 10.0);
    mat.insert(1, 0, 20.0);
    mat.insert(2, 0, 30.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 10.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 1), 20.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 2), 30.0);
}

TEST(AccumulateSingleRow, OutOfRangeThrows) {
    AosMat mat(3, 5);
    EXPECT_THROW(accumulate(mat, 3), std::out_of_range);
    EXPECT_THROW(accumulate(mat, 100), std::out_of_range);
}

TEST(AccumulateSingleRow, NegativeValues) {
    AosMat mat(1, 5);
    mat.insert(0, 0, -3.0);
    mat.insert(0, 1, 5.0);
    mat.insert(0, 2, -2.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 0.0);
}

TEST(AccumulateSingleRow, OverwrittenElementUsesLatestValue) {
    AosMat mat(1, 5);
    mat.insert(0, 0, 10.0);
    mat.insert(0, 0, 25.0);
    mat.lock();
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 25.0);
}

// ─────────────────────────────────────────────────────
// Whole-matrix accumulate
// ─────────────────────────────────────────────────────

TEST(AccumulateWholeMatrix, EmptyMatrix) {
    AosMat mat(3, 5);
    auto result = accumulate(mat);
    ASSERT_EQ(result.size(), 3u);
    for (auto v : result) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(AccumulateWholeMatrix, EachRowAccumulatedIndependently) {
    AosMat mat(3, 5);
    mat.insert(0, 0, 1.0);
    mat.insert(0, 1, 2.0);

    mat.insert(1, 0, 10.0);

    mat.insert(2, 0, 100.0);
    mat.insert(2, 1, 200.0);
    mat.insert(2, 2, 300.0);

    auto result = accumulate(mat);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_DOUBLE_EQ(result[0], 3.0);
    EXPECT_DOUBLE_EQ(result[1], 10.0);
    EXPECT_DOUBLE_EQ(result[2], 600.0);
}

TEST(AccumulateWholeMatrix, SingleRowMatrix) {
    AosMat mat(1, 3);
    mat.insert(0, 0, 7.0);
    mat.insert(0, 1, 3.0);

    auto result = accumulate(mat);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0], 10.0);
}

TEST(AccumulateWholeMatrix, ResultSizeMatchesRowCount) {
    AosMat mat(50, 10);
    auto result = accumulate(mat);
    EXPECT_EQ(result.size(), 50u);
}

// ─────────────────────────────────────────────────────
// Open and locked mode both work
// ─────────────────────────────────────────────────────

TEST(AccumulateMode, WorksInOpenMode) {
    AosMat mat(1, 5);
    mat.insert(0, 0, 1.0);
    mat.insert(0, 1, 2.0);
    mat.insert(0, 2, 3.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 6.0);
}

TEST(AccumulateMode, WorksInLockedMode) {
    AosMat mat(1, 5);
    mat.insert(0, 0, 1.0);
    mat.insert(0, 1, 2.0);
    mat.lock();
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 3.0);
}

TEST(AccumulateMode, OpenAndLockedGiveSameResult) {
    AosMat mat(2, 5);
    mat.insert(0, 0, 1.0);
    mat.insert(1, 0, 5.0);
    auto open_result = accumulate(mat);
    mat.lock();
    auto locked_result = accumulate(mat);
    EXPECT_DOUBLE_EQ(open_result[0], locked_result[0]);
    EXPECT_DOUBLE_EQ(open_result[1], locked_result[1]);
}

// ─────────────────────────────────────────────────────
// Layout variants
// ─────────────────────────────────────────────────────

TEST(AccumulateLayouts, SoaLayout) {
    SoaMat mat(2, 5);
    mat.insert(0, 0, 4.0);
    mat.insert(0, 1, 6.0);
    mat.insert(1, 0, 1.0);

    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 10.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 1), 1.0);
}

TEST(AccumulateLayouts, SoaWholeMatrix) {
    SoaMat mat(2, 5);
    mat.insert(0, 0, 4.0);
    mat.insert(0, 1, 6.0);
    mat.insert(1, 0, 1.0);

    auto result = accumulate(mat);
    EXPECT_DOUBLE_EQ(result[0], 10.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
}

// ─────────────────────────────────────────────────────
// Value types
// ─────────────────────────────────────────────────────

TEST(AccumulateValueTypes, FloatPrecision) {
    SoaMatF mat(1, 4);
    mat.insert(0, 0, 0.1f);
    mat.insert(0, 1, 0.2f);
    mat.insert(0, 2, 0.3f);
    EXPECT_NEAR(accumulate(mat, 0), 0.6f, 1e-6f);
}

TEST(AccumulateValueTypes, IntegerType) {
    AosMatI mat(1, 5);
    mat.insert(0, 0, 10);
    mat.insert(0, 1, 20);
    mat.insert(0, 2, 30);
    EXPECT_EQ(accumulate(mat, 0), 60);
}

TEST(AccumulateValueTypes, IntegerWholeMatrix) {
    AosMatI mat(2, 3);
    mat.insert(0, 0, 1);
    mat.insert(0, 1, 2);
    mat.insert(1, 0, 100);

    auto result = accumulate(mat);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 100);
}

// ─────────────────────────────────────────────────────
// Dense / stress
// ─────────────────────────────────────────────────────

TEST(AccumulateStress, FullyDenseRow) {
    const uint32_t cols = 50; // keep under default BufferN=64 for open-mode test
    AosMat mat(1, cols);

    double expected = 0.0;
    for (uint32_t c = 0; c < cols; ++c) {
        double val = static_cast<double>(c + 1);
        mat.insert(0, c, val);
        expected += val;
    }

    EXPECT_DOUBLE_EQ(accumulate(mat, 0), expected);
}

TEST(AccumulateStress, ManyRows) {
    const uint32_t rows = 500;
    const uint32_t cols = 10;
    AosMat mat(rows, cols);

    for (uint32_t r = 0; r < rows; ++r) {
        mat.insert(r, 0, static_cast<double>(r));
    }

    auto result = accumulate(mat);
    ASSERT_EQ(result.size(), rows);
    for (uint32_t r = 0; r < rows; ++r) {
        EXPECT_DOUBLE_EQ(result[r], static_cast<double>(r));
    }
}

// ─────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────

TEST(AccumulateEdge, ZeroValueInsertions) {
    AosMat mat(1, 5);
    mat.insert(0, 0, 0.0);
    mat.insert(0, 1, 0.0);
    mat.insert(0, 2, 5.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 5.0);
}

TEST(AccumulateEdge, AllZeroValues) {
    AosMat mat(1, 5);
    mat.insert(0, 0, 0.0);
    mat.insert(0, 1, 0.0);
    mat.lock();
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 0.0);
}

TEST(AccumulateEdge, LargeValues) {
    AosMat mat(1, 3);
    mat.insert(0, 0, 1e15);
    mat.insert(0, 1, 1e15);
    mat.insert(0, 2, 1e15);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 3e15);
}

TEST(AccumulateEdge, MixedPositiveNegativeCancels) {
    AosMat mat(1, 4);
    mat.insert(0, 0, 100.0);
    mat.insert(0, 1, -50.0);
    mat.insert(0, 2, -50.0);
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 0.0);
}

TEST(AccumulateEdge, RepeatedOverwrites) {
    AosMat mat(1, 3);
    for (int i = 0; i < 100; ++i) {
        mat.insert(0, 0, static_cast<double>(i));
    }
    mat.lock();
    // final value at (0,0) should be 99 (last write wins after dedup)
    EXPECT_DOUBLE_EQ(accumulate(mat, 0), 99.0);
}

} // anonymous namespace
