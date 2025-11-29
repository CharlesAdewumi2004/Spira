#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

#include <../include/spira/spira.hpp>

namespace soaTests
{

    using Index = std::size_t;
    using Value = double;
    using TestLayoutTag = spira::layout::tags::soa_tag;

    using Matrix = spira::matrix<TestLayoutTag, Index, Value>;

    namespace
    {

        TEST(SpmvTest, EmptyMatrixProducesZeroVector)
        {
            Matrix A(3, 3); // 3x3, but no entries

            std::vector<Value> x = {1.0, 2.0, 3.0};
            std::vector<Value> y(3, -1.0); // will be overwritten

            spira::algorithms::spmv(A, x, y);

            ASSERT_EQ(y.size(), 3u);
            EXPECT_DOUBLE_EQ(y[0], 0.0);
            EXPECT_DOUBLE_EQ(y[1], 0.0);
            EXPECT_DOUBLE_EQ(y[2], 0.0);
        }

        TEST(SpmvTest, SingleNonZeroElement)
        {
            Matrix A(3, 3);
            // A(1,2) = 5
            A.add(1, 2, 5.0);

            std::vector<Value> x = {1.0, 2.0, 3.0};
            std::vector<Value> y(3, 0.0);

            spira::algorithms::spmv(A, x, y);

            EXPECT_DOUBLE_EQ(y[0], 0.0);
            EXPECT_DOUBLE_EQ(y[1], 5.0 * 3.0); // 5 * x[2]
            EXPECT_DOUBLE_EQ(y[2], 0.0);
        }

        TEST(SpmvTest, SingleRowMultipleNonZeros)
        {
            Matrix A(1, 5);

            // Row 0: [1, 0, 2, 0, 3]
            A.add(0, 0, 1.0);
            A.add(0, 2, 2.0);
            A.add(0, 4, 3.0);

            std::vector<Value> x = {10.0, 20.0, 30.0, 40.0, 50.0};
            std::vector<Value> y(1, 0.0);

            spira::algorithms::spmv(A, x, y);

            // y0 = 1*10 + 2*30 + 3*50 = 220
            ASSERT_EQ(y.size(), 1u);
            EXPECT_DOUBLE_EQ(y[0], 1.0 * 10.0 + 2.0 * 30.0 + 3.0 * 50.0);
        }

        TEST(SpmvTest, MultiRowSparseMatrix)
        {
            Matrix A(3, 4);

            // Row 0: [2, 0, 1, 0]
            A.add(0, 0, 2.0);
            A.add(0, 2, 1.0);

            // Row 1: [0, 3, 0, 4]
            A.add(1, 1, 3.0);
            A.add(1, 3, 4.0);

            // Row 2: [0, 0, 0, 5]
            A.add(2, 3, 5.0);

            std::vector<Value> x = {1.0, 2.0, 3.0, 4.0};
            std::vector<Value> y(3, 0.0);

            spira::algorithms::spmv(A, x, y);

            EXPECT_DOUBLE_EQ(y[0], 2.0 * 1.0 + 1.0 * 3.0); // 5
            EXPECT_DOUBLE_EQ(y[1], 3.0 * 2.0 + 4.0 * 4.0); // 22
            EXPECT_DOUBLE_EQ(y[2], 5.0 * 4.0);             // 20
        }

        TEST(SpmvTest, ExplicitZeroIsIgnored)
        {
            Matrix A(2, 3);

            // This should be removed/ignored by your ValueTraits::is_zero logic:
            A.add(0, 1, 0.0);
            // Only this one should actually contribute:
            A.add(1, 2, 4.0);

            std::vector<Value> x = {10.0, 20.0, 30.0};
            std::vector<Value> y(2, 0.0);

            spira::algorithms::spmv(A, x, y);

            EXPECT_DOUBLE_EQ(y[0], 0.0);
            EXPECT_DOUBLE_EQ(y[1], 4.0 * 30.0);
        }

        // -----------------------
        // Dimension mismatch tests
        // -----------------------

        TEST(SpmvTest, ThrowsWhenXTooShort)
        {
            Matrix A(3, 3);
            A.add(0, 0, 1.0);

            std::vector<Value> x = {1.0, 2.0}; // too small
            std::vector<Value> y(3, 0.0);

            EXPECT_THROW(
                spira::algorithms::spmv(A, x, y),
                std::invalid_argument);
        }

        TEST(SpmvTest, ThrowsWhenXTooLong)
        {
            Matrix A(3, 3);
            std::vector<Value> x = {1.0, 2.0, 3.0, 4.0}; // too big
            std::vector<Value> y(3, 0.0);

            EXPECT_THROW(
                spira::algorithms::spmv(A, x, y),
                std::invalid_argument);
        }

        TEST(SpmvTest, ThrowsWhenYTooShort)
        {
            Matrix A(3, 3);
            std::vector<Value> x = {1.0, 2.0, 3.0};
            std::vector<Value> y = {0.0, 0.0}; // too small

            EXPECT_THROW(
                spira::algorithms::spmv(A, x, y),
                std::invalid_argument);
        }

        TEST(SpmvTest, ThrowsWhenYTooLong)
        {
            Matrix A(3, 3);
            std::vector<Value> x = {1.0, 2.0, 3.0};
            std::vector<Value> y = {0.0, 0.0, 0.0, 0.0}; // too big

            EXPECT_THROW(
                spira::algorithms::spmv(A, x, y),
                std::invalid_argument);
        }

        TEST(SpmvTest, ThrowsWhenBothXAndYWrongSizes)
        {
            Matrix A(2, 4); // 2x4 matrix, so x.size() must be 4, y.size() must be 2

            std::vector<Value> x = {1.0, 2.0, 3.0}; // wrong (3)
            std::vector<Value> y = {0.0, 0.0, 0.0}; // wrong (3)

            EXPECT_THROW(
                spira::algorithms::spmv(A, x, y),
                std::invalid_argument);
        }

        TEST(SpmvTest, ThrowsWhenXEmptyButMatrixHasColumns)
        {
            Matrix A(2, 3);
            A.add(0, 0, 1.0);

            std::vector<Value> x; // empty
            std::vector<Value> y(2, 0.0);

            EXPECT_THROW(
                spira::algorithms::spmv(A, x, y),
                std::invalid_argument);
        }

        TEST(SpmvTest, ThrowsWhenYEmptyButMatrixHasRows)
        {
            Matrix A(2, 3);
            A.add(0, 0, 1.0);

            std::vector<Value> x = {1.0, 2.0, 3.0};
            std::vector<Value> y; // empty

            EXPECT_THROW(
                spira::algorithms::spmv(A, x, y),
                std::invalid_argument);
        }

        // 0x0 matrix with empty x,y should be OK (if your impl supports it)
        TEST(SpmvTest, ZeroByZeroMatrixWithEmptyVectorsIsFine)
        {
            Matrix A(0, 0);

            std::vector<Value> x;
            std::vector<Value> y;

            // x.size() == n_cols() == 0
            // y.size() == n_rows() == 0
            EXPECT_NO_THROW(spira::algorithms::spmv(A, x, y));
            EXPECT_TRUE(y.empty());
        }

    } // anonymous namespace
}