// tests/hash_map_buffer_tests.cpp
//
// Coverage for spira::buffer::impls::hash_map_buffer (the buffer the
// benchmarks use).
//
// Two regressions are pinned here:
//   1. The buffer's sort_and_dedup() was once missing, so the CRTP base
//      forwarded to itself and lock() recursed until the stack ran out.
//   2. contains()/get() consulted only the open-mode hash map, so entries
//      materialised by sort_and_dedup() were invisible to reads.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <spira/matrix/matrix.hpp>

using Layout = spira::layout::tags::soa_tag;
using BufTag = spira::buffer::tags::hash_map_buffer;

using HMat = spira::matrix<Layout, uint32_t, double, BufTag, 64>;

// ─────────────────────────────────────────────────────────────────────────────
// lock() -> flat CSR -> read back
// ─────────────────────────────────────────────────────────────────────────────

TEST(HashMapBuffer, LockBuildsCsrAndReadsBack)
{
    HMat A(4, 4);
    A.insert(0, 1u, 2.0);
    A.insert(2, 3u, 5.0);
    A.lock();

    EXPECT_EQ(A.nnz(), 2u);
    EXPECT_DOUBLE_EQ(A.get(0, 1u), 2.0);
    EXPECT_DOUBLE_EQ(A.get(2, 3u), 5.0);
    EXPECT_TRUE(A.contains(2, 3u));
    EXPECT_FALSE(A.contains(1, 1u));
    EXPECT_DOUBLE_EQ(A.get(1, 1u), 0.0);
}

TEST(HashMapBuffer, LastWriteWinsWithinOpenPhase)
{
    HMat A(4, 4);
    A.insert(0, 1u, 2.0);
    A.insert(0, 1u, 7.0);

    EXPECT_DOUBLE_EQ(A.get(0, 1u), 7.0); // open mode reads the hash map

    A.lock();
    EXPECT_EQ(A.nnz(), 1u);
    EXPECT_DOUBLE_EQ(A.get(0, 1u), 7.0);
}

TEST(HashMapBuffer, ZeroValueFilteredOnFirstLock)
{
    HMat A(4, 4);
    A.insert(0, 1u, 0.0);
    A.insert(0, 2u, 3.0);
    A.lock();

    EXPECT_EQ(A.nnz(), 1u);
    EXPECT_FALSE(A.contains(0, 1u));
    EXPECT_DOUBLE_EQ(A.get(0, 2u), 3.0);
}

// Zeros must survive sort_and_dedup() so the re-lock can read them as
// deletion signals against the committed CSR.
TEST(HashMapBuffer, ZeroInsertDeletesCommittedEntry)
{
    HMat A(4, 4);
    A.insert(0, 1u, 2.0);
    A.insert(0, 2u, 3.0);
    A.lock();
    ASSERT_EQ(A.nnz(), 2u);

    A.open();
    A.insert(0, 1u, 0.0); // delete column 1
    A.lock();

    EXPECT_EQ(A.nnz(), 1u);
    EXPECT_FALSE(A.contains(0, 1u));
    EXPECT_DOUBLE_EQ(A.get(0, 1u), 0.0);
    EXPECT_DOUBLE_EQ(A.get(0, 2u), 3.0);
}

TEST(HashMapBuffer, RepeatedLockCyclesMergeIntoCsr)
{
    HMat A(8, 8);
    for (uint32_t r = 0; r < 8; ++r)
        A.insert(r, r, 1.0);
    A.lock();
    ASSERT_EQ(A.nnz(), 8u);

    for (uint32_t cycle = 1; cycle <= 3; ++cycle)
    {
        A.open();
        A.insert(cycle, 7u, cycle * 10.0);
        A.lock();
    }

    EXPECT_EQ(A.nnz(), 11u); // 8 diagonal + 3 new (1,7) (2,7) (3,7)
    for (uint32_t r = 0; r < 8; ++r)
        EXPECT_DOUBLE_EQ(A.get(r, r), 1.0) << "diagonal row " << r;
    EXPECT_DOUBLE_EQ(A.get(1, 7u), 10.0);
    EXPECT_DOUBLE_EQ(A.get(2, 7u), 20.0);
    EXPECT_DOUBLE_EQ(A.get(3, 7u), 30.0);
}

TEST(HashMapBuffer, OverwriteCommittedValueAcrossLockCycle)
{
    HMat A(4, 4);
    A.insert(1, 2u, 4.0);
    A.lock();

    A.open();
    A.insert(1, 2u, 9.0); // buffer wins on collision
    A.lock();

    EXPECT_EQ(A.nnz(), 1u);
    EXPECT_DOUBLE_EQ(A.get(1, 2u), 9.0);
}

TEST(HashMapBuffer, AccumulateOpenAndLocked)
{
    HMat A(4, 4);
    A.insert(0, 1u, 2.0);
    A.insert(0, 3u, 5.0);

    EXPECT_DOUBLE_EQ(A.accumulate(0), 7.0); // open: hash map
    A.lock();
    EXPECT_DOUBLE_EQ(A.accumulate(0), 7.0); // locked: CSR slice
    EXPECT_DOUBLE_EQ(A.accumulate(1), 0.0);
}
