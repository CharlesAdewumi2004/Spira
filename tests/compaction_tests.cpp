// tests/compaction_tests.cpp
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <spira/matrix/csr_build.hpp>
#include <spira/matrix/row.hpp>

using Layout = spira::layout::tags::aos_tag;
using I      = std::size_t;
using V      = double;
using RowT   = spira::row<Layout, I, V>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static RowT make_row(std::size_t col_limit,
                     std::initializer_list<std::pair<I, V>> entries)
{
    RowT r(col_limit);
    for (auto [c, v] : entries)
        r.insert(c, v);
    r.lock();
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(BuildCSR, EmptyRows_OffsetsAllZero)
{
    std::vector<RowT> rows;
    rows.push_back(make_row(4, {}));
    rows.push_back(make_row(4, {}));
    rows.push_back(make_row(4, {}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.n_rows, 3u);
    ASSERT_EQ(csr.nnz,    0u);

    EXPECT_EQ(csr.offsets[0], 0u);
    EXPECT_EQ(csr.offsets[1], 0u);
    EXPECT_EQ(csr.offsets[2], 0u);
    EXPECT_EQ(csr.offsets[3], 0u);

    // pairs pointer is null when nnz == 0
    EXPECT_EQ(csr.pairs.get(), nullptr);
}

TEST(BuildCSR, SingleEntry_CorrectOffsetColVal)
{
    std::vector<RowT> rows;
    rows.push_back(make_row(6, {{2, 7.5}}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.n_rows, 1u);
    ASSERT_EQ(csr.nnz,    1u);

    EXPECT_EQ(csr.offsets[0], 0u);
    EXPECT_EQ(csr.offsets[1], 1u);

    EXPECT_EQ(csr.pairs.get()[0].column, I{2});
    EXPECT_DOUBLE_EQ(csr.pairs.get()[0].value, 7.5);
}

TEST(BuildCSR, MultipleRowsMultipleEntries_OffsetsColsValsExact)
{
    // 3 rows, column limit 5
    //   row 0: (0,1.0), (3,2.0)   → 2 entries
    //   row 1: (1,3.0)             → 1 entry
    //   row 2: (2,4.0)             → 1 entry
    // Expected offsets: [0, 2, 3, 4]
    // Expected pairs:   [{0,1.0}, {3,2.0}, {1,3.0}, {2,4.0}]
    std::vector<RowT> rows;
    rows.push_back(make_row(5, {{0, 1.0}, {3, 2.0}}));
    rows.push_back(make_row(5, {{1, 3.0}}));
    rows.push_back(make_row(5, {{2, 4.0}}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.n_rows, 3u);
    ASSERT_EQ(csr.nnz,    4u);

    EXPECT_EQ(csr.offsets[0], 0u);
    EXPECT_EQ(csr.offsets[1], 2u);
    EXPECT_EQ(csr.offsets[2], 3u);
    EXPECT_EQ(csr.offsets[3], 4u);

    const auto *pairs = csr.pairs.get();

    EXPECT_EQ(pairs[0].column, I{0});  EXPECT_DOUBLE_EQ(pairs[0].value, 1.0);
    EXPECT_EQ(pairs[1].column, I{3});  EXPECT_DOUBLE_EQ(pairs[1].value, 2.0);
    EXPECT_EQ(pairs[2].column, I{1});  EXPECT_DOUBLE_EQ(pairs[2].value, 3.0);
    EXPECT_EQ(pairs[3].column, I{2});  EXPECT_DOUBLE_EQ(pairs[3].value, 4.0);
}

TEST(BuildCSR, SparseRows_FirstAndLastEmpty)
{
    // row 0: empty
    // row 1: (0,9.0), (4,8.0)
    // row 2: empty
    // Expected offsets: [0, 0, 2, 2]
    std::vector<RowT> rows;
    rows.push_back(make_row(5, {}));
    rows.push_back(make_row(5, {{0, 9.0}, {4, 8.0}}));
    rows.push_back(make_row(5, {}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.n_rows, 3u);
    ASSERT_EQ(csr.nnz,    2u);

    EXPECT_EQ(csr.offsets[0], 0u);
    EXPECT_EQ(csr.offsets[1], 0u);
    EXPECT_EQ(csr.offsets[2], 2u);
    EXPECT_EQ(csr.offsets[3], 2u);

    EXPECT_EQ(csr.pairs.get()[0].column, I{0});  EXPECT_DOUBLE_EQ(csr.pairs.get()[0].value, 9.0);
    EXPECT_EQ(csr.pairs.get()[1].column, I{4});  EXPECT_DOUBLE_EQ(csr.pairs.get()[1].value, 8.0);
}

TEST(BuildCSR, ColsAreSortedPerRow)
{
    // Insertion order is reversed; after lock() the buffer is sorted ascending.
    // row 0: insert (4,4.0) then (1,1.0) then (2,2.0)
    // After lock: sorted as (1,1.0), (2,2.0), (4,4.0)
    std::vector<RowT> rows;
    rows.push_back(make_row(6, {{4, 4.0}, {1, 1.0}, {2, 2.0}}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.nnz, 3u);
    EXPECT_EQ(csr.pairs.get()[0].column, I{1});
    EXPECT_EQ(csr.pairs.get()[1].column, I{2});
    EXPECT_EQ(csr.pairs.get()[2].column, I{4});
}

TEST(BuildCSR, ZeroRows_IsBuilt)
{
    std::vector<RowT> rows;  // empty vector

    auto csr = spira::build_csr<Layout>(rows);

    EXPECT_EQ(csr.n_rows, 0u);
    EXPECT_EQ(csr.nnz,    0u);
    EXPECT_TRUE(csr.is_built());  // offsets[0] allocated
}
