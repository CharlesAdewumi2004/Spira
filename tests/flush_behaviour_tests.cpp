#include <spira/spira.hpp>
#include <gtest/gtest.h>
#include <cstddef>

// Tests for slab+buffer merge semantics via lock()/open().

// ── Helpers ──────────────────────────────────────────────────────────────────

template <class Mat>
static void insert_all(Mat& mat, std::initializer_list<std::tuple<std::size_t, std::size_t, double>> xs)
{
    for (auto [r, c, v] : xs) mat.insert(r, c, v);
}

// ── Single open→lock cycle ────────────────────────────────────────────────────

TEST(LockBehaviorTest, SingleLock_DedupLastWriteWins_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(1, 10);

    mat.insert(0, 3, 3.0);
    mat.insert(0, 1, 1.0);
    mat.insert(0, 1, 9.0); // overwrite col 1 — 9.0 should win

    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 9.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 3), 3.0);
}

TEST(LockBehaviorTest, SingleLock_DedupLastWriteWins_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(1, 10);

    mat.insert(0, 1, 1.0);
    mat.insert(0, 2, 2.0);
    mat.insert(0, 3, 3.0);
    mat.insert(0, 4, 4.0);
    mat.insert(0, 1, 5.0); // overwrite col 1

    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(0, 1), 5.0);
    EXPECT_TRUE(mat.contains(0, 2));
    EXPECT_TRUE(mat.contains(0, 3));
    EXPECT_TRUE(mat.contains(0, 4));
    EXPECT_EQ(mat.row_nnz(0), 4u);
    EXPECT_EQ(mat.nnz(), 4u);
}

// ── Idempotent lock ──────────────────────────────────────────────────────────

TEST(LockBehaviorTest, LockIsIdempotent_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(2, 2);

    mat.insert(0, 0, 7.7);
    mat.insert(0, 1, 8.8);

    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 7.7);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 8.8);

    // Locking an already-locked matrix is a no-op.
    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 8.8);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(LockBehaviorTest, LockIsIdempotent_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(2, 2);

    mat.insert(0, 0, 7.7);
    mat.insert(0, 1, 8.8);

    mat.lock();
    mat.lock(); // second lock is no-op

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 8.8);
    EXPECT_EQ(mat.nnz(), 2u);
}

// ── Open → lock → open → insert → lock (incremental merge) ──────────────────

TEST(LockBehaviorTest, IncrementalMerge_SlabPreserved_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(1, 10);

    // Cycle 1
    mat.insert(0, 3, 4.0);
    mat.insert(0, 5, 99.0);
    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 2u);

    // Cycle 2: add new column, existing slab preserved
    mat.open();
    mat.insert(0, 9, 7.0);
    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 3u);
    EXPECT_DOUBLE_EQ(mat.get(0, 3), 4.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 5), 99.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 9), 7.0);
}

TEST(LockBehaviorTest, IncrementalMerge_SlabPreserved_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(1, 10);

    mat.insert(0, 3, 4.0);
    mat.insert(0, 5, 99.0);
    mat.lock();

    mat.open();
    mat.insert(0, 9, 7.0);
    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 3u);
    EXPECT_DOUBLE_EQ(mat.get(0, 3), 4.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 5), 99.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 9), 7.0);
}

// ── Buffer overwrites existing slab entry ────────────────────────────────────

TEST(LockBehaviorTest, BufferOverwritesSlabEntry_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(1, 10);

    // Cycle 1: commit col 5 = 2.0 to slab
    mat.insert(0, 5, 2.0);
    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(0, 5), 2.0);

    // Cycle 2: overwrite col 5 via buffer — buffer wins
    mat.open();
    mat.insert(0, 5, 99.0);
    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(0, 5), 99.0);
    EXPECT_EQ(mat.row_nnz(0), 1u);
}

TEST(LockBehaviorTest, BufferOverwritesSlabEntry_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(1, 10);

    mat.insert(0, 5, 2.0);
    mat.lock();

    mat.open();
    mat.insert(0, 5, 99.0);
    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(0, 5), 99.0);
    EXPECT_EQ(mat.row_nnz(0), 1u);
}

// ── Zero values filtered during lock ────────────────────────────────────────

TEST(LockBehaviorTest, ZeroValueFilteredDuringLock_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(1, 10);

    mat.insert(0, 1, 5.0);
    mat.insert(0, 2, 0.0); // explicit zero
    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 2), 0.0); // returns zero (not present)
    EXPECT_FALSE(mat.contains(0, 2));
}

TEST(LockBehaviorTest, ZeroValueFilteredDuringLock_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(1, 10);

    mat.insert(0, 1, 5.0);
    mat.insert(0, 2, 0.0);
    mat.lock();

    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 5.0);
    EXPECT_FALSE(mat.contains(0, 2));
}

// ── Reads in open mode use buffer-first then slab ────────────────────────────

TEST(LockBehaviorTest, OpenModeGet_BufferFirst_ThenSlab) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(1, 10);

    // Commit col 7 = 1.0 to slab
    mat.insert(0, 7, 1.0);
    mat.lock();

    // In open mode, col 7 is in slab. New insert (col 7 = 9.0) goes to buffer.
    mat.open();
    mat.insert(0, 7, 9.0);

    // Before lock: get should see buffer value (9.0) not slab value (1.0)
    EXPECT_DOUBLE_EQ(mat.get(0, 7), 9.0);

    mat.lock();
    EXPECT_DOUBLE_EQ(mat.get(0, 7), 9.0);
}
