#include <spira/spira.hpp>
#include <gtest/gtest.h>
#include <cstddef>

// ------------------ InsertHeavy -> Balanced ------------------

TEST(ModeSwitchTest, InsertHeavyToBalanced_AOS)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    mat.add(0, 0, 5.0);
    mat.flush();
    mat.add(1, 1, 7.5);

    mat.set_mode(spira::mode::matrix_mode::balanced);

    EXPECT_TRUE(mat.contains(0, 0));
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 5.0);
    EXPECT_TRUE(mat.contains(1, 1));
    EXPECT_DOUBLE_EQ(mat.get(1, 1), 7.5);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(1), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, InsertHeavyToBalanced_SOA)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    mat.add(0, 0, 5.0);
    mat.flush();
    mat.add(1, 1, 7.5);

    mat.set_mode(spira::mode::matrix_mode::balanced);

    EXPECT_TRUE(mat.contains(0, 0));
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 5.0);
    EXPECT_TRUE(mat.contains(1, 1));
    EXPECT_DOUBLE_EQ(mat.get(1, 1), 7.5);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(1), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

// ------------------ Balanced -> InsertHeavy ------------------

TEST(ModeSwitchTest, BalancedToInsertHeavy_AOS)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::balanced);

    mat.add(0, 1, 2.5);
    mat.flush();
    mat.add(2, 2, 3.5);

    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    EXPECT_TRUE(mat.contains(0, 1));
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 2.5);
    EXPECT_TRUE(mat.contains(2, 2));
    EXPECT_DOUBLE_EQ(mat.get(2, 2), 3.5);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(2), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, BalancedToInsertHeavy_SOA)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::balanced);

    mat.add(0, 1, 2.5);
    mat.flush();
    mat.add(2, 2, 3.5);

    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    EXPECT_TRUE(mat.contains(0, 1));
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 2.5);
    EXPECT_TRUE(mat.contains(2, 2));
    EXPECT_DOUBLE_EQ(mat.get(2, 2), 3.5);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(2), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

// ------------------ Balanced -> SpMV ------------------

TEST(ModeSwitchTest, BalancedToSpMV_AOS)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::balanced);

    mat.add(0, 3, 1.5);
    mat.flush();
    mat.add(1, 0, 2.5);

    mat.set_mode(spira::mode::matrix_mode::spmv);

    EXPECT_TRUE(mat.contains(0, 3));
    EXPECT_DOUBLE_EQ(mat.get(0, 3), 1.5);
    EXPECT_TRUE(mat.contains(1, 0));
    EXPECT_DOUBLE_EQ(mat.get(1, 0), 2.5);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(1), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, BalancedToSpMV_SOA)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::balanced);

    mat.add(0, 3, 1.5);
    mat.flush();
    mat.add(1, 0, 2.5);

    mat.set_mode(spira::mode::matrix_mode::spmv);

    EXPECT_TRUE(mat.contains(0, 3));
    EXPECT_DOUBLE_EQ(mat.get(0, 3), 1.5);
    EXPECT_TRUE(mat.contains(1, 0));
    EXPECT_DOUBLE_EQ(mat.get(1, 0), 2.5);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(1), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

// ------------------ SpMV -> Balanced ------------------

TEST(ModeSwitchTest, SpMVToBalanced_AOS)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::spmv);

    mat.add(0, 0, 10.0);
    mat.flush(); // committed
    mat.add(2, 1, 4.0);

    mat.set_mode(spira::mode::matrix_mode::balanced);

    EXPECT_TRUE(mat.contains(0, 0));
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 10.0);
    EXPECT_TRUE(mat.contains(2, 1));
    EXPECT_DOUBLE_EQ(mat.get(2, 1), 4.0);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(2), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, SpMVToBalanced_SOA)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::spmv);

    mat.add(0, 0, 10.0);
    mat.flush();
    mat.add(2, 1, 4.0);

    mat.set_mode(spira::mode::matrix_mode::balanced);

    EXPECT_TRUE(mat.contains(0, 0));
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 10.0);
    EXPECT_TRUE(mat.contains(2, 1));
    EXPECT_DOUBLE_EQ(mat.get(2, 1), 4.0);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(0), 1u);
    EXPECT_EQ(mat.row_nnz(2), 1u);
    EXPECT_EQ(mat.nnz(), 2u);
}

// ------------------ InsertHeavy -> SpMV ------------------

TEST(ModeSwitchTest, InsertHeavyToSpMV_AOS)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    mat.add(1, 2, 8.0);
    mat.flush();
    mat.add(1, 3, 9.0);

    mat.set_mode(spira::mode::matrix_mode::spmv);

    EXPECT_TRUE(mat.contains(1, 2));
    EXPECT_DOUBLE_EQ(mat.get(1, 2), 8.0);
    EXPECT_TRUE(mat.contains(1, 3));
    EXPECT_DOUBLE_EQ(mat.get(1, 3), 9.0);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(1), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, InsertHeavyToSpMV_SOA)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    mat.add(1, 2, 8.0);
    mat.flush();
    mat.add(1, 3, 9.0);

    mat.set_mode(spira::mode::matrix_mode::spmv);

    EXPECT_TRUE(mat.contains(1, 2));
    EXPECT_DOUBLE_EQ(mat.get(1, 2), 8.0);
    EXPECT_TRUE(mat.contains(1, 3));
    EXPECT_DOUBLE_EQ(mat.get(1, 3), 9.0);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(1), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}

// ------------------ SpMV -> InsertHeavy ------------------

TEST(ModeSwitchTest, SpMVToInsertHeavy_AOS)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::spmv);

    mat.add(3, 3, 11.0);
    mat.flush();
    mat.add(3, 2, 12.0);

    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    EXPECT_TRUE(mat.contains(3, 3));
    EXPECT_DOUBLE_EQ(mat.get(3, 3), 11.0);
    EXPECT_TRUE(mat.contains(3, 2));
    EXPECT_DOUBLE_EQ(mat.get(3, 2), 12.0);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(3), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(ModeSwitchTest, SpMVToInsertHeavy_SOA)
{
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(4, 4);
    mat.set_mode(spira::mode::matrix_mode::spmv);

    mat.add(3, 3, 11.0);
    mat.flush();
    mat.add(3, 2, 12.0);

    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    EXPECT_TRUE(mat.contains(3, 3));
    EXPECT_DOUBLE_EQ(mat.get(3, 3), 11.0);
    EXPECT_TRUE(mat.contains(3, 2));
    EXPECT_DOUBLE_EQ(mat.get(3, 2), 12.0);

    mat.flush();
    EXPECT_EQ(mat.row_nnz(3), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}
