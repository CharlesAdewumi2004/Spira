#include <spira/spira.hpp>
#include <gtest/gtest.h>
#include <cstddef>

// Tests for open ↔ locked mode transitions and multi-cycle slab accumulation.

TEST(ModeSwitchTest, OpenToLock_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);

    mat.insert(0, 0, 5.0);
    mat.insert(1, 1, 7.5);
    mat.lock();

    EXPECT_TRUE(mat.is_locked());
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(mat.get(1, 1), 7.5);
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(1), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, OpenToLock_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);

    mat.insert(0, 0, 5.0);
    mat.insert(1, 1, 7.5);
    mat.lock();

    EXPECT_TRUE(mat.is_locked());
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 5.0);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, LockToOpen_SlabPreserved_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);

    mat.insert(0, 3, 1.5);
    mat.lock();
    mat.open();

    EXPECT_FALSE(mat.is_locked());
    EXPECT_TRUE(mat.contains(0, 3));
    EXPECT_DOUBLE_EQ(mat.get(0, 3), 1.5);
}

TEST(ModeSwitchTest, LockToOpen_SlabPreserved_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);

    mat.insert(0, 3, 1.5);
    mat.lock();
    mat.open();

    EXPECT_FALSE(mat.is_locked());
    EXPECT_TRUE(mat.contains(0, 3));
    EXPECT_DOUBLE_EQ(mat.get(0, 3), 1.5);
}

TEST(ModeSwitchTest, MultiCycle_DataAccumulates_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);

    mat.insert(0, 0, 5.0);
    mat.lock();

    mat.open();
    mat.insert(1, 1, 7.5);
    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(mat.get(1, 1), 7.5);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, MultiCycle_DataAccumulates_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);

    mat.insert(0, 3, 1.5);
    mat.lock();

    mat.open();
    mat.insert(1, 0, 2.5);
    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(0, 3), 1.5);
    EXPECT_DOUBLE_EQ(mat.get(1, 0), 2.5);
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(1), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, MultiCycle_BufferOverwritesSlab_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);

    mat.insert(1, 2, 8.0);
    mat.lock();

    mat.open();
    mat.insert(1, 2, 99.0); // overwrite slab entry
    mat.insert(1, 3, 9.0);
    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(1, 2), 99.0);
    EXPECT_DOUBLE_EQ(mat.get(1, 3), 9.0);
    EXPECT_EQ(mat.row_nnz(1), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, MultiCycle_BufferOverwritesSlab_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);

    mat.insert(3, 3, 11.0);
    mat.lock();

    mat.open();
    mat.insert(3, 3, 99.0);
    mat.insert(3, 2, 12.0);
    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(3, 3), 99.0);
    EXPECT_DOUBLE_EQ(mat.get(3, 2), 12.0);
    EXPECT_EQ(mat.row_nnz(3), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, OpenModeInsertsVisibleBeforeLock_AOS) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);

    mat.insert(0, 0, 10.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 10.0);
    EXPECT_TRUE(mat.contains(0, 0));

    mat.insert(2, 1, 4.0);
    EXPECT_DOUBLE_EQ(mat.get(2, 1), 4.0);

    mat.lock();

    EXPECT_DOUBLE_EQ(mat.get(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(mat.get(2, 1), 4.0);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, OpenModeInsertsVisibleBeforeLock_SOA) {
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);

    mat.insert(0, 0, 10.0);
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 10.0);

    mat.lock();
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 10.0);
    EXPECT_EQ(mat.nnz(), 1u);
}
