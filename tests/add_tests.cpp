#include <spira/spira.hpp>
#include <gtest/gtest.h>

#include <stdexcept>

// matrix::add() must behave the same whichever buffer backs the rows, so every
// case runs against each layout / buffer combination.

template <class Layout, class BufTag>
struct AddConfig
{
    using mat = spira::matrix<Layout, uint32_t, double, BufTag, 64>;
};

using AosArray = AddConfig<spira::layout::tags::aos_tag,
                           spira::buffer::tags::array_buffer<spira::layout::tags::aos_tag>>;
using SoaArray = AddConfig<spira::layout::tags::soa_tag,
                           spira::buffer::tags::array_buffer<spira::layout::tags::soa_tag>>;
using SoaHash = AddConfig<spira::layout::tags::soa_tag,
                          spira::buffer::tags::hash_map_buffer>;
using AosHash = AddConfig<spira::layout::tags::aos_tag,
                          spira::buffer::tags::hash_map_buffer>;

template <class Config>
class MatrixAddTest : public ::testing::Test
{
protected:
    using mat = typename Config::mat;
};

using AddConfigs = ::testing::Types<AosArray, SoaArray, SoaHash, AosHash>;
TYPED_TEST_SUITE(MatrixAddTest, AddConfigs);

TYPED_TEST(MatrixAddTest, AddToMissingEntryActsAsInsert)
{
    typename TestFixture::mat m(4, 4);
    m.add(1, 2, 3.5);
    EXPECT_DOUBLE_EQ(m.get(1, 2), 3.5);
    m.lock();
    EXPECT_EQ(m.nnz(), 1u);
    EXPECT_DOUBLE_EQ(m.get(1, 2), 3.5);
}

TYPED_TEST(MatrixAddTest, AddsWithinOneCycleSum)
{
    typename TestFixture::mat m(4, 4);
    m.add(0, 0, 2.0);
    m.add(0, 0, 3.0);
    m.add(0, 0, -0.5);
    EXPECT_DOUBLE_EQ(m.get(0, 0), 4.5);
    m.lock();
    EXPECT_EQ(m.row_nnz(0), 1u);
    EXPECT_DOUBLE_EQ(m.get(0, 0), 4.5);
}

TYPED_TEST(MatrixAddTest, AddOntoCommittedValue)
{
    typename TestFixture::mat m(4, 4);
    m.insert(2, 1, 5.0);
    m.insert(2, 3, 7.0);
    m.lock();

    m.open();
    m.add(2, 1, 1.0);
    m.add(2, 1, 1.0);
    EXPECT_DOUBLE_EQ(m.get(2, 1), 7.0);
    m.lock();

    EXPECT_EQ(m.row_nnz(2), 2u);
    EXPECT_DOUBLE_EQ(m.get(2, 1), 7.0);
    EXPECT_DOUBLE_EQ(m.get(2, 3), 7.0); // untouched neighbour survives the merge

    // A third cycle keeps building on the merged value.
    m.open();
    m.add(2, 1, 3.0);
    m.lock();
    EXPECT_DOUBLE_EQ(m.get(2, 1), 10.0);
}

TYPED_TEST(MatrixAddTest, InsertAfterAddOverwrites)
{
    typename TestFixture::mat m(4, 4);
    m.add(3, 3, 2.0);
    m.add(3, 3, 2.0);
    m.insert(3, 3, 9.0);
    m.lock();
    EXPECT_DOUBLE_EQ(m.get(3, 3), 9.0);
}

TYPED_TEST(MatrixAddTest, AddAfterInsertBuildsOnInsertedValue)
{
    typename TestFixture::mat m(4, 4);
    m.insert(3, 0, 9.0);
    m.add(3, 0, 1.0);
    m.lock();
    EXPECT_DOUBLE_EQ(m.get(3, 0), 10.0);
}

TYPED_TEST(MatrixAddTest, SumToZeroDeletesCommittedEntry)
{
    typename TestFixture::mat m(4, 4);
    m.insert(1, 1, 4.0);
    m.insert(1, 2, 6.0);
    m.lock();
    ASSERT_EQ(m.nnz(), 2u);

    m.open();
    m.add(1, 1, -4.0);
    m.lock();

    EXPECT_EQ(m.nnz(), 1u);
    EXPECT_FALSE(m.contains(1, 1));
    EXPECT_DOUBLE_EQ(m.get(1, 2), 6.0);
}

TYPED_TEST(MatrixAddTest, SumToZeroWithinFirstCycleIsNotStored)
{
    typename TestFixture::mat m(4, 4);
    m.add(0, 3, 2.5);
    m.add(0, 3, -2.5);
    m.lock();
    EXPECT_EQ(m.nnz(), 0u);
    EXPECT_FALSE(m.contains(0, 3));
}

TYPED_TEST(MatrixAddTest, AddFeedsSpmv)
{
    // Assemble a 1D Laplacian-style stencil from overlapping element
    // contributions, the pattern add() exists for.
    constexpr uint32_t N = 5;
    typename TestFixture::mat m(N, N);
    for (uint32_t e = 0; e + 1 < N; ++e)
    {
        m.add(e, e, 1.0);
        m.add(e + 1, e + 1, 1.0);
        m.add(e, e + 1, -1.0);
        m.add(e + 1, e, -1.0);
    }
    m.lock();

    EXPECT_DOUBLE_EQ(m.get(0, 0), 1.0);
    for (uint32_t i = 1; i + 1 < N; ++i)
        EXPECT_DOUBLE_EQ(m.get(i, i), 2.0) << "row " << i;
    EXPECT_DOUBLE_EQ(m.get(N - 1, N - 1), 1.0);

    // Every row of this operator sums to zero.
    const std::vector<double> x(N, 1.0);
    const std::vector<double> y = m * x;
    for (uint32_t i = 0; i < N; ++i)
        EXPECT_NEAR(y[i], 0.0, 1e-12) << "row " << i;
}

TYPED_TEST(MatrixAddTest, ThrowsWhenLocked)
{
    typename TestFixture::mat m(2, 2);
    m.lock();
    EXPECT_THROW(m.add(0, 0, 1.0), std::logic_error);
}

TYPED_TEST(MatrixAddTest, ThrowsOnOutOfRangeIndex)
{
    typename TestFixture::mat m(2, 2);
    EXPECT_THROW(m.add(2, 0, 1.0), std::out_of_range);
    EXPECT_THROW(m.add(0, 2, 1.0), std::out_of_range);
}
