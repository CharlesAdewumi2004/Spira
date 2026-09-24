// tests/compaction_tests.cpp
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/matrix/storage/csr_build.hpp>
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

template <class Csr>
static I col_at(const Csr &csr, std::size_t row, std::size_t k)
{
    return csr.pairs.get()[csr.row_start[row] + k].column;
}

template <class Csr>
static V val_at(const Csr &csr, std::size_t row, std::size_t k)
{
    return csr.pairs.get()[csr.row_start[row] + k].value;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_csr
// ─────────────────────────────────────────────────────────────────────────────

TEST(BuildCSR, EmptyRows_NoSlotsAssigned)
{
    std::vector<RowT> rows;
    rows.push_back(make_row(4, {}));
    rows.push_back(make_row(4, {}));
    rows.push_back(make_row(4, {}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.n_rows, 3u);
    ASSERT_EQ(csr.nnz,    0u);
    EXPECT_EQ(csr.end,    0u);
    for (std::size_t r = 0; r < 3; ++r)
    {
        EXPECT_EQ(csr.row_len[r], 0u);
        EXPECT_EQ(csr.row_cap[r], 0u); // empty rows get no slack
    }

    // pairs pointer is null when nothing is allocated
    EXPECT_EQ(csr.pairs.get(), nullptr);
}

TEST(BuildCSR, SingleEntry_CorrectSlotColVal)
{
    std::vector<RowT> rows;
    rows.push_back(make_row(6, {{2, 7.5}}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.n_rows, 1u);
    ASSERT_EQ(csr.nnz,    1u);

    EXPECT_EQ(csr.row_start[0], 0u);
    EXPECT_EQ(csr.row_len[0],   1u);
    EXPECT_EQ(csr.row_cap[0],   spira::row_slot_capacity(1));

    EXPECT_EQ(col_at(csr, 0, 0), I{2});
    EXPECT_DOUBLE_EQ(val_at(csr, 0, 0), 7.5);
}

TEST(BuildCSR, MultipleRows_SlotsInRowOrderWithSlack)
{
    //   row 0: (0,1.0), (3,2.0)   → 2 entries
    //   row 1: (1,3.0)             → 1 entry
    //   row 2: (2,4.0)             → 1 entry
    std::vector<RowT> rows;
    rows.push_back(make_row(5, {{0, 1.0}, {3, 2.0}}));
    rows.push_back(make_row(5, {{1, 3.0}}));
    rows.push_back(make_row(5, {{2, 4.0}}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.n_rows, 3u);
    ASSERT_EQ(csr.nnz,    4u);

    EXPECT_EQ(csr.row_len[0], 2u);
    EXPECT_EQ(csr.row_len[1], 1u);
    EXPECT_EQ(csr.row_len[2], 1u);

    EXPECT_EQ(csr.row_start[0], 0u);
    EXPECT_EQ(csr.row_start[1], csr.row_cap[0]);
    EXPECT_EQ(csr.row_start[2], csr.row_cap[0] + csr.row_cap[1]);
    EXPECT_EQ(csr.end, csr.row_cap[0] + csr.row_cap[1] + csr.row_cap[2]);
    for (std::size_t r = 0; r < 3; ++r)
        EXPECT_GT(csr.row_cap[r], csr.row_len[r]) << "row " << r << " has no slack";
    EXPECT_GE(csr.capacity, csr.end);

    EXPECT_EQ(col_at(csr, 0, 0), I{0});  EXPECT_DOUBLE_EQ(val_at(csr, 0, 0), 1.0);
    EXPECT_EQ(col_at(csr, 0, 1), I{3});  EXPECT_DOUBLE_EQ(val_at(csr, 0, 1), 2.0);
    EXPECT_EQ(col_at(csr, 1, 0), I{1});  EXPECT_DOUBLE_EQ(val_at(csr, 1, 0), 3.0);
    EXPECT_EQ(col_at(csr, 2, 0), I{2});  EXPECT_DOUBLE_EQ(val_at(csr, 2, 0), 4.0);
}

TEST(BuildCSR, ZeroValuesAreNotStored)
{
    std::vector<RowT> rows;
    rows.push_back(make_row(5, {{0, 9.0}, {2, 0.0}, {4, 8.0}}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.nnz, 2u);
    EXPECT_EQ(col_at(csr, 0, 0), I{0});
    EXPECT_EQ(col_at(csr, 0, 1), I{4});
}

TEST(BuildCSR, ColsAreSortedPerRow)
{
    std::vector<RowT> rows;
    rows.push_back(make_row(6, {{4, 4.0}, {1, 1.0}, {2, 2.0}}));

    auto csr = spira::build_csr<Layout>(rows);

    ASSERT_EQ(csr.nnz, 3u);
    EXPECT_EQ(col_at(csr, 0, 0), I{1});
    EXPECT_EQ(col_at(csr, 0, 1), I{2});
    EXPECT_EQ(col_at(csr, 0, 2), I{4});
}

TEST(BuildCSR, ZeroRows_IsBuilt)
{
    std::vector<RowT> rows;  // empty vector

    auto csr = spira::build_csr<Layout>(rows);

    EXPECT_EQ(csr.n_rows, 0u);
    EXPECT_EQ(csr.nnz,    0u);
    EXPECT_TRUE(csr.is_built());
}

TEST(RowSlotCapacity, SlackPolicy)
{
    EXPECT_EQ(spira::row_slot_capacity(0), 0u);
    EXPECT_EQ(spira::row_slot_capacity(1), 2u);
    EXPECT_EQ(spira::row_slot_capacity(3), 4u);
    EXPECT_EQ(spira::row_slot_capacity(13), 16u);
    EXPECT_EQ(spira::row_slot_capacity(100), 125u);
}

// ─────────────────────────────────────────────────────────────────────────────
// relock_rows
// ─────────────────────────────────────────────────────────────────────────────

namespace
{
    // Rows 0..n-1, row r holding columns {0, 2, 4, ...} (4 entries each) with
    // value r + 1, laid out by build_csr and with slices installed.
    struct built_rows
    {
        std::vector<RowT> rows;
        spira::csr_storage<Layout, I, V> csr;

        explicit built_rows(std::size_t n)
        {
            for (std::size_t r = 0; r < n; ++r)
                rows.push_back(make_row(64, {{0, r + 1.0}, {2, r + 1.0}, {4, r + 1.0}, {6, r + 1.0}}));
            csr = spira::build_csr<Layout>(rows);
            spira::install_slices<Layout>(csr, rows);
            for (auto &row : rows)
                row.clear_buffer_content();
        }

        void edit(std::size_t r, std::initializer_list<std::pair<I, V>> entries)
        {
            rows[r].open();
            for (auto [c, v] : entries)
                rows[r].insert(c, v);
        }
    };
}

TEST(RelockRows, EditThatFitsStaysInItsSlot)
{
    built_rows b(4);
    const auto starts_before = std::vector<std::size_t>(b.csr.row_start.get(), b.csr.row_start.get() + 4);
    const V *row2_val = b.rows[2].get(0);
    ASSERT_NE(row2_val, nullptr);

    b.edit(1, {{3, 9.0}}); // 5 entries, slot holds row_slot_capacity(4) = 5
    ASSERT_LE(5u, b.csr.row_cap[1]);
    const bool repacked = spira::relock_rows<Layout>(b.csr, b.rows, {1});

    EXPECT_FALSE(repacked);
    for (std::size_t r = 0; r < 4; ++r)
        EXPECT_EQ(b.csr.row_start[r], starts_before[r]) << "row " << r << " moved";
    EXPECT_EQ(b.csr.holes, 0u);
    EXPECT_EQ(b.csr.row_len[1], 5u);
    EXPECT_EQ(b.csr.nnz, 17u);

    ASSERT_TRUE(b.rows[1].contains(3));
    EXPECT_DOUBLE_EQ(*b.rows[1].get(3), 9.0);
    // Untouched rows still read through the very same memory.
    EXPECT_EQ(b.rows[2].get(0), row2_val);
}

TEST(RelockRows, OverflowMovesOnlyThatRowToTheTail)
{
    // 64 rows x 5 slots leaves a free tail of 40 slots (1/8 of 320), enough
    // for one moved row; a tiny matrix would repack instead.
    built_rows b(64);
    const std::size_t end_before = b.csr.end;
    const std::size_t cap_before = b.csr.row_cap[1];
    const V *row0_val = b.rows[0].get(0);
    const V *row3_val = b.rows[63].get(6);

    b.edit(1, {{1, 5.0}, {3, 5.0}, {5, 5.0}}); // 7 entries > 5 slots
    const bool repacked = spira::relock_rows<Layout>(b.csr, b.rows, {1});

    ASSERT_FALSE(repacked) << "the free tail should absorb one moved row";
    EXPECT_EQ(b.csr.row_start[1], end_before);
    EXPECT_GE(b.csr.row_cap[1], 7u);
    EXPECT_EQ(b.csr.holes, cap_before);
    EXPECT_EQ(b.csr.row_len[1], 7u);

    for (I c = 0; c < 7; ++c)
        EXPECT_DOUBLE_EQ(*b.rows[1].get(c), c % 2 == 0 ? 2.0 : 5.0) << "col " << c;
    EXPECT_EQ(b.rows[0].get(0), row0_val);
    EXPECT_EQ(b.rows[63].get(6), row3_val);
}

TEST(RelockRows, DeletionShrinksRowInPlace)
{
    built_rows b(3);
    const std::size_t start_before = b.csr.row_start[1];

    b.edit(1, {{2, 0.0}, {6, 0.0}});
    spira::relock_rows<Layout>(b.csr, b.rows, {1});

    EXPECT_EQ(b.csr.row_start[1], start_before);
    EXPECT_EQ(b.csr.row_len[1], 2u);
    EXPECT_EQ(b.csr.nnz, 10u);
    EXPECT_FALSE(b.rows[1].contains(2));
    EXPECT_FALSE(b.rows[1].contains(6));
    EXPECT_DOUBLE_EQ(*b.rows[1].get(0), 2.0);
    EXPECT_DOUBLE_EQ(*b.rows[1].get(4), 2.0);
}

TEST(RelockRows, EmptyRowGetsItsFirstSlotInTheTail)
{
    std::vector<RowT> rows;
    rows.push_back(make_row(8, {{0, 1.0}, {1, 1.0}, {2, 1.0}, {3, 1.0}, {4, 1.0}, {5, 1.0}, {6, 1.0}, {7, 1.0}}));
    rows.push_back(make_row(8, {}));
    auto csr = spira::build_csr<Layout>(rows);
    spira::install_slices<Layout>(csr, rows);
    ASSERT_EQ(csr.row_cap[1], 0u);

    rows[1].open();
    rows[1].insert(4, 2.5);
    spira::relock_rows<Layout>(csr, rows, {1});

    EXPECT_EQ(csr.row_len[1], 1u);
    EXPECT_DOUBLE_EQ(*rows[1].get(4), 2.5);
    EXPECT_DOUBLE_EQ(*rows[0].get(7), 1.0);
}

TEST(RelockRows, RepackWhenTheTailRunsOut)
{
    built_rows b(8);
    // Grow every row past its slot at once: more than the free tail holds.
    std::vector<std::size_t> dirty;
    for (std::size_t r = 0; r < 8; ++r)
    {
        b.edit(r, {{1, 7.0}, {3, 7.0}, {5, 7.0}, {7, 7.0}});
        dirty.push_back(r);
    }
    const bool repacked = spira::relock_rows<Layout>(b.csr, b.rows, dirty);

    ASSERT_TRUE(repacked);
    EXPECT_EQ(b.csr.holes, 0u);
    EXPECT_EQ(b.csr.nnz, 64u);
    std::size_t expected_start = 0;
    for (std::size_t r = 0; r < 8; ++r)
    {
        EXPECT_EQ(b.csr.row_start[r], expected_start) << "repack lays rows out in order";
        expected_start += b.csr.row_cap[r];
        for (I c = 0; c < 8; ++c)
            EXPECT_DOUBLE_EQ(*b.rows[r].get(c), c % 2 == 0 ? r + 1.0 : 7.0) << "row " << r << " col " << c;
    }
}

TEST(RelockRows, RepeatedGrowthStaysCorrectAcrossRepacks)
{
    // One row grows by an entry per cycle: it moves to the tail, the tail
    // eventually runs out and the CSR repacks, and the values stay right.
    built_rows b(16);
    for (I c = 8; c < 60; ++c)
    {
        b.edit(5, {{c, static_cast<V>(c)}});
        spira::relock_rows<Layout>(b.csr, b.rows, {5});
        ASSERT_EQ(b.csr.row_len[5], 4u + (c - 7)) << "after adding col " << c;
    }
    for (I c = 8; c < 60; ++c)
        EXPECT_DOUBLE_EQ(*b.rows[5].get(c), static_cast<V>(c));
    EXPECT_DOUBLE_EQ(*b.rows[5].get(0), 6.0);
    EXPECT_DOUBLE_EQ(*b.rows[15].get(6), 16.0);
    EXPECT_EQ(b.csr.nnz, 16u * 4u + 52u);
}

// ─────────────────────────────────────────────────────────────────────────────
// matrix::lock() only touches edited rows
// ─────────────────────────────────────────────────────────────────────────────

TEST(MatrixRelock, CleanRowsKeepTheirMemoryAcrossCycles)
{
    spira::matrix<Layout, uint32_t, double> m(100, 100);
    for (uint32_t r = 0; r < 100; ++r)
        for (uint32_t c = 0; c < 5; ++c)
            m.insert(r, (r + c) % 100, 1.0);
    m.lock();

    std::vector<const double *> before(100);
    for (uint32_t r = 0; r < 100; ++r)
        before[r] = m.row_at(r).get(r);

    m.open();
    m.add(40, 40, 1.0);
    m.insert(41, 99, 3.0);
    m.lock();

    EXPECT_DOUBLE_EQ(m.get(40, 40), 2.0);
    EXPECT_DOUBLE_EQ(m.get(41, 99), 3.0);
    for (uint32_t r = 0; r < 100; ++r)
    {
        if (r != 40 && r != 41)
        {
            EXPECT_EQ(m.row_at(r).get(r), before[r]) << "row " << r << " was rewritten";
        }
    }
}

TEST(MatrixRelock, OpenThenLockWithoutEditsChangesNothing)
{
    spira::matrix<Layout, uint32_t, double> m(10, 10);
    for (uint32_t r = 0; r < 10; ++r)
        m.insert(r, r, r + 1.0);
    m.lock();
    const auto end_before = m.csr()->end;
    const double *p = m.row_at(7).get(7);

    m.open();
    EXPECT_TRUE(m.is_open());
    m.lock();

    EXPECT_EQ(m.csr()->end, end_before);
    EXPECT_EQ(m.row_at(7).get(7), p);
    EXPECT_EQ(m.nnz(), 10u);
}

TEST(MatrixRelock, ClearDropsOnlyPendingEdits)
{
    spira::matrix<Layout, uint32_t, double> m(4, 4);
    m.insert(0, 0, 1.0);
    m.lock();
    m.open();
    m.insert(1, 1, 2.0);
    m.clear();
    m.lock();
    EXPECT_DOUBLE_EQ(m.get(0, 0), 1.0);
    EXPECT_FALSE(m.contains(1, 1));
}

TEST(MatrixCopy, CopyOwnsItsCsr)
{
    // Regression: the defaulted copy constructor left the copy's rows pointing
    // into the source's CSR, so reads after the source died were a
    // use-after-free.
    using M = spira::matrix<Layout, uint32_t, double>;
    auto src = std::make_unique<M>(2, 2);
    src->insert(0, 0, 1.5);
    src->insert(1, 1, 2.5);
    src->lock();

    M copy(*src);
    EXPECT_NE(copy.row_at(0).get(0), src->row_at(0).get(0));

    M assigned(1, 1);
    assigned = *src;
    src.reset();

    EXPECT_DOUBLE_EQ(copy.get(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(copy.get(1, 1), 2.5);
    EXPECT_DOUBLE_EQ(assigned.get(1, 1), 2.5);

    copy.open();
    copy.insert(0, 1, 4.0);
    copy.lock();
    EXPECT_DOUBLE_EQ(copy.get(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(copy.get(0, 0), 1.5);
}
