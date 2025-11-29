#include <gtest/gtest.h>

#include "../include/spira/spira.hpp"

namespace soaTests
{

    // ============================================================================
    // TYPE ALIASES / CONFIG
    // ============================================================================

    namespace
    {

        using index_t = std::size_t;
        using value_t = double;

        using TestLayoutTag = spira::layout::tags::soa_tag;

        using Row = spira::row<TestLayoutTag, index_t, value_t>;
        using Matrix = spira::matrix<TestLayoutTag, index_t, value_t>;

    } // namespace

    // ============================================================================
    // ROW TESTS
    // ============================================================================

    TEST(RowBasicTests, DefaultConstructedRowIsEmpty)
    {
        Row r;

        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_GE(r.capacity(), 0u);
    }

    TEST(RowBasicTests, ReserveAndClearBehavesCorrectly)
    {
        Row r;
        r.reserve(10);

        EXPECT_GE(r.capacity(), 10u);
        EXPECT_TRUE(r.empty());

        // Fake add via set_row to ensure size changes
        std::vector<std::pair<index_t, value_t>> elems = {
            {0, 1.0},
            {3, 2.0}};

        // column_limit_ is set via ctor with hint + limit
        Row r2(/*reserve_hint*/ 4, /*column_limit*/ 5);
        r2.set_row(elems);

        EXPECT_FALSE(r2.empty());
        EXPECT_EQ(r2.size(), 2u);

        r2.clear();
        EXPECT_TRUE(r2.empty());
        EXPECT_EQ(r2.size(), 0u);
    }

    TEST(RowAddRemoveTests, AddWithinLimitsInsertsElement)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 10);

        r.add(3, 4.5);

        EXPECT_FALSE(r.empty());
        EXPECT_TRUE(r.contains(3));

        auto ptr = r.get(3);
        ASSERT_NE(ptr, nullptr);
        EXPECT_DOUBLE_EQ(*ptr, 4.5);
    }

    TEST(RowAddRemoveTests, AddBeyondColumnLimitIsIgnored)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 5);

        r.add(10, 1.23); // out of range
        EXPECT_TRUE(r.empty());
        EXPECT_FALSE(r.contains(10));
        EXPECT_EQ(r.get(10), nullptr);
    }

    TEST(RowAddRemoveTests, AddingZeroValueDoesNotInsert)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 10);

        value_t zero = spira::traits::ValueTraits<value_t>::zero();
        r.add(2, zero);

        EXPECT_TRUE(r.empty());
        EXPECT_FALSE(r.contains(2));
        EXPECT_EQ(r.get(2), nullptr);
    }

    TEST(RowAddRemoveTests, OverwriteExistingValue)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 10);

        r.add(2, 1.0);
        r.add(2, 3.5); // overwrite

        EXPECT_TRUE(r.contains(2));
        auto ptr = r.get(2);
        ASSERT_NE(ptr, nullptr);
        EXPECT_DOUBLE_EQ(*ptr, 3.5);
    }

    TEST(RowAddRemoveTests, AddZeroOnExistingKeyRemovesIt)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 10);

        r.add(2, 1.0);
        EXPECT_TRUE(r.contains(2));

        value_t zero = spira::traits::ValueTraits<value_t>::zero();
        r.add(2, zero); // should erase

        EXPECT_FALSE(r.contains(2));
        EXPECT_EQ(r.get(2), nullptr);
        EXPECT_TRUE(r.empty());
    }

    TEST(RowAddRemoveTests, RemoveExistingKeyErasesEntry)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 10);

        r.add(4, 2.0);
        EXPECT_TRUE(r.contains(4));

        r.remove(4);
        EXPECT_FALSE(r.contains(4));
        EXPECT_EQ(r.get(4), nullptr);
        EXPECT_TRUE(r.empty());
    }

    TEST(RowAddRemoveTests, RemoveNonExistingKeyIsNoOp)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 10);

        r.add(1, 5.0);
        EXPECT_TRUE(r.contains(1));
        EXPECT_FALSE(r.contains(2));

        r.remove(2); // no-op
        EXPECT_TRUE(r.contains(1));
        EXPECT_EQ(r.size(), 1u);
    }

    TEST(RowAddRemoveTests, RemoveBeyondColumnLimitIsIgnored)
    {
        Row r(/*reserve_hint*/ 4, /*column_limit*/ 5);

        r.add(2, 1.0);
        EXPECT_TRUE(r.contains(2));

        r.remove(100); // beyond limit
        EXPECT_TRUE(r.contains(2));
        EXPECT_EQ(r.size(), 1u);
    }

    // ============================================================================
    // set_row TESTS
    // ============================================================================

    TEST(RowSetRowTests, SetRowBasicUsage)
    {
        Row r(/*reserve_hint*/ 8, /*column_limit*/ 10);

        std::vector<std::pair<index_t, value_t>> elems = {
            {2, 4.0},
            {5, 7.0}};

        r.set_row(elems);

        EXPECT_EQ(r.size(), 2u);
        EXPECT_TRUE(r.contains(2));
        EXPECT_TRUE(r.contains(5));

        auto v2 = r.get(2);
        auto v5 = r.get(5);

        ASSERT_NE(v2, nullptr);
        ASSERT_NE(v5, nullptr);
        EXPECT_DOUBLE_EQ(*v2, 4.0);
        EXPECT_DOUBLE_EQ(*v5, 7.0);
    }

    TEST(RowSetRowTests, SetRowFiltersOutOfRangeAndZeroValues)
    {
        Row r(/*reserve_hint*/ 8, /*column_limit*/ 5);

        value_t zero = spira::traits::ValueTraits<value_t>::zero();

        std::vector<std::pair<index_t, value_t>> elems = {
            {1, 3.0}, // valid
            {6, 4.0}, // out of column_limit_
            {2, zero} // valid column but zero -> should be ignored
        };

        r.set_row(elems);

        EXPECT_EQ(r.size(), 1u);
        EXPECT_TRUE(r.contains(1));
        EXPECT_FALSE(r.contains(6));
        EXPECT_FALSE(r.contains(2));

        auto v1 = r.get(1);
        ASSERT_NE(v1, nullptr);
        EXPECT_DOUBLE_EQ(*v1, 3.0);
    }

    TEST(RowSetRowTests, SetRowDeduplicatesByLastValue)
    {
        Row r(/*reserve_hint*/ 8, /*column_limit*/ 10);

        std::vector<std::pair<index_t, value_t>> elems = {
            {3, 1.0},
            {3, 2.0},
            {3, 5.0}, // last one should win
            {7, 10.0}};

        r.set_row(elems);

        EXPECT_EQ(r.size(), 2u);
        EXPECT_TRUE(r.contains(3));
        EXPECT_TRUE(r.contains(7));

        auto v3 = r.get(3);
        auto v7 = r.get(7);

        ASSERT_NE(v3, nullptr);
        ASSERT_NE(v7, nullptr);
        EXPECT_DOUBLE_EQ(*v3, 5.0); // last write wins
        EXPECT_DOUBLE_EQ(*v7, 10.0);
    }

    TEST(RowSetRowTests, ContainsAndGetRespectColumnLimit)
    {
        Row r(/*reserve_hint*/ 8, /*column_limit*/ 5);

        std::vector<std::pair<index_t, value_t>> elems = {
            {0, 1.0},
            {4, 2.0},
            {5, 3.0} // out of range, dropped
        };

        r.set_row(elems);

        EXPECT_TRUE(r.contains(0));
        EXPECT_TRUE(r.contains(4));
        EXPECT_FALSE(r.contains(5));

        EXPECT_NE(r.get(0), nullptr);
        EXPECT_NE(r.get(4), nullptr);
        EXPECT_EQ(r.get(5), nullptr);

        // Contains/get on col >= limit_ must be false/nullptr
        EXPECT_FALSE(r.contains(10));
        EXPECT_EQ(r.get(10), nullptr);
    }

    // ============================================================================
    // MATRIX TESTS
    // ============================================================================

    TEST(MatrixBasicTests, ConstructAndShape)
    {
        Matrix m(/*row_limit*/ 4, /*column_limit*/ 6);

        auto shape = m.get_shape();
        EXPECT_EQ(shape.first, 4u);
        EXPECT_EQ(shape.second, 6u);

        EXPECT_EQ(m.n_rows(), 4u);
        EXPECT_EQ(m.n_cols(), 6u);
        EXPECT_TRUE(m.empty());
        EXPECT_EQ(m.nnz(), 0u);

        for (index_t r = 0; r < m.n_rows(); ++r)
        {
            EXPECT_EQ(m.row_nnz(r), 0u);
        }
    }

    TEST(MatrixBasicTests, RowNnzThrowsOnOutOfRange)
    {
        Matrix m(3, 5);

        EXPECT_THROW(m.row_nnz(3), std::out_of_range);
        EXPECT_THROW(m.row_nnz(100), std::out_of_range);
    }

    TEST(MatrixAddGetTests, AddAndGetSingleElement)
    {
        Matrix m(3, 3);

        m.add(1, 2, 4.0);

        EXPECT_FALSE(m.empty());
        EXPECT_EQ(m.nnz(), 1u);
        EXPECT_EQ(m.row_nnz(0), 0u);
        EXPECT_EQ(m.row_nnz(1), 1u);
        EXPECT_EQ(m.row_nnz(2), 0u);

        EXPECT_TRUE(m.contains(1, 2));
        EXPECT_DOUBLE_EQ(m.get(1, 2), 4.0);
    }

    TEST(MatrixAddGetTests, AddThrowsOnOutOfRangeRowOrCol)
    {
        Matrix m(2, 2);

        EXPECT_THROW(m.add(2, 0, 1.0), std::out_of_range);
        EXPECT_THROW(m.add(0, 2, 1.0), std::out_of_range);
    }

    TEST(MatrixAddGetTests, GetThrowsOnOutOfRangeRowOrCol)
    {
        Matrix m(2, 2);

        EXPECT_THROW(m.get(2, 0), std::out_of_range);
        EXPECT_THROW(m.get(0, 2), std::out_of_range);
    }

    TEST(MatrixAddGetTests, GetReturnsZeroForMissingEntry)
    {
        Matrix m(3, 3);

        value_t zero = spira::traits::ValueTraits<value_t>::zero();

        // No entries yet
        EXPECT_DOUBLE_EQ(m.get(0, 0), zero);

        m.add(1, 1, 5.0);
        EXPECT_DOUBLE_EQ(m.get(1, 1), 5.0);
        EXPECT_DOUBLE_EQ(m.get(1, 2), zero); // missing entry
    }

    TEST(MatrixMutationTests, ClearRemovesAllEntries)
    {
        Matrix m(3, 3);

        m.add(0, 0, 1.0);
        m.add(1, 1, 2.0);
        m.add(2, 2, 3.0);

        EXPECT_FALSE(m.empty());
        EXPECT_EQ(m.nnz(), 3u);

        m.clear();

        EXPECT_TRUE(m.empty());
        EXPECT_EQ(m.nnz(), 0u);
        for (index_t r = 0; r < m.n_rows(); ++r)
        {
            EXPECT_EQ(m.row_nnz(r), 0u);
        }

        value_t zero = spira::traits::ValueTraits<value_t>::zero();
        EXPECT_DOUBLE_EQ(m.get(0, 0), zero);
        EXPECT_DOUBLE_EQ(m.get(1, 1), zero);
        EXPECT_DOUBLE_EQ(m.get(2, 2), zero);
    }

    TEST(MatrixRowSetTests, SetRowDelegatesToRowCorrectly)
    {
        Matrix m(3, 5);

        std::vector<std::pair<index_t, value_t>> elems = {
            {0, 1.0},
            {3, 2.0}};

        m.set_row(1, elems);

        EXPECT_EQ(m.row_nnz(1), 2u);
        EXPECT_TRUE(m.contains(1, 0));
        EXPECT_TRUE(m.contains(1, 3));
        EXPECT_FALSE(m.contains(1, 4));

        EXPECT_DOUBLE_EQ(m.get(1, 0), 1.0);
        EXPECT_DOUBLE_EQ(m.get(1, 3), 2.0);
    }

    TEST(MatrixRowSetTests, SetRowThrowsOnOutOfRangeRow)
    {
        Matrix m(2, 5);
        std::vector<std::pair<index_t, value_t>> elems = {
            {1, 1.0}};

        EXPECT_THROW(m.set_row(2, elems), std::out_of_range);
    }

    TEST(MatrixContainsRemoveTests, ContainsAndRemoveWork)
    {
        Matrix m(3, 4);

        m.add(0, 1, 1.0);
        m.add(0, 2, 2.0);
        m.add(2, 3, 3.0);

        EXPECT_TRUE(m.contains(0, 1));
        EXPECT_TRUE(m.contains(0, 2));
        EXPECT_TRUE(m.contains(2, 3));
        EXPECT_FALSE(m.contains(1, 1));

        m.remove(0, 1);
        EXPECT_FALSE(m.contains(0, 1));
        EXPECT_EQ(m.row_nnz(0), 1u);
        EXPECT_EQ(m.nnz(), 2u);

        m.remove(2, 3);
        EXPECT_FALSE(m.contains(2, 3));
        EXPECT_EQ(m.row_nnz(2), 0u);
        EXPECT_EQ(m.nnz(), 1u);
    }

    TEST(MatrixContainsRemoveTests, RemoveThrowsOnOutOfRangeRowOrCol)
    {
        Matrix m(2, 2);

        EXPECT_THROW(m.remove(2, 0), std::out_of_range);
        EXPECT_THROW(m.remove(0, 2), std::out_of_range);
    }

    TEST(MatrixContainsRemoveTests, ContainsThrowsOnOutOfRangeRowOrCol)
    {
        Matrix m(2, 2);

        EXPECT_THROW(m.contains(2, 0), std::out_of_range);
        EXPECT_THROW(m.contains(0, 2), std::out_of_range);
    }
}