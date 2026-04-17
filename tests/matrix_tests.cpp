#include <spira/spira.hpp>
#include <gtest/gtest.h>

// ------------------ Basic insertion / get / contains ------------------

TEST(MatrixBasicTest, InsertAndContains_AOS)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(5, 5);

    EXPECT_TRUE(mat.empty());
    EXPECT_EQ(mat.nnz(), 0u);
    EXPECT_EQ(mat.row_nnz(0), 0u);

    V v12 = 3.14159;
    mat.insert(1, 2, v12);

    EXPECT_FALSE(mat.empty());
    EXPECT_EQ(mat.nnz(), 1u);
    EXPECT_EQ(mat.row_nnz(1), 1u);
    EXPECT_EQ(mat.row_nnz(0), 0u);

    EXPECT_TRUE(mat.contains(1, 2));
    EXPECT_DOUBLE_EQ(mat.get(1, 2), v12);

    EXPECT_FALSE(mat.contains(1, 3));
    EXPECT_DOUBLE_EQ(mat.get(1, 3), 0.0);
    EXPECT_FALSE(mat.contains(0, 0));
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 0.0);

    V v23 = 2.71828;
    mat.insert(2, 3, v23);

    EXPECT_EQ(mat.nnz(), 2u);
    EXPECT_EQ(mat.row_nnz(2), 1u);
    EXPECT_TRUE(mat.contains(2, 3));
    EXPECT_DOUBLE_EQ(mat.get(2, 3), v23);
    EXPECT_FALSE(mat.contains(2, 2));
    EXPECT_DOUBLE_EQ(mat.get(2, 2), 0.0);
}

TEST(MatrixBasicTest, InsertAndContains_SOA)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(5, 5);

    EXPECT_TRUE(mat.empty());
    EXPECT_EQ(mat.nnz(), 0u);

    V v12 = 3.14159;
    mat.insert(1, 2, v12);

    EXPECT_TRUE(mat.contains(1, 2));
    EXPECT_DOUBLE_EQ(mat.get(1, 2), v12);
    EXPECT_EQ(mat.row_nnz(1), 1u);

    V v23 = 2.71828;
    mat.insert(2, 3, v23);
    EXPECT_TRUE(mat.contains(2, 3));
    EXPECT_DOUBLE_EQ(mat.get(2, 3), v23);

    EXPECT_EQ(mat.nnz(), 2u);
}

// ------------------ Overwrite semantics (last write wins) ------------------

TEST(MatrixBasicTest, Overwrite_LastWriteWins_AOS)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(3, 3);

    I r = 1, c = 1;

    mat.insert(r, c, 5.0);
    EXPECT_DOUBLE_EQ(mat.get(r, c), 5.0);
    EXPECT_TRUE(mat.contains(r, c));
    EXPECT_EQ(mat.row_nnz(r), 1u);

    mat.insert(r, c, 7.5);
    EXPECT_DOUBLE_EQ(mat.get(r, c), 7.5);
    EXPECT_TRUE(mat.contains(r, c));

    mat.lock();
    EXPECT_TRUE(mat.contains(r, c));
    EXPECT_DOUBLE_EQ(mat.get(r, c), 7.5);

    EXPECT_EQ(mat.row_nnz(r), 1u);
    EXPECT_EQ(mat.nnz(), 1u);
}

TEST(MatrixBasicTest, Overwrite_LastWriteWins_SOA)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(3, 3);

    I r = 1, c = 1;
    mat.insert(r, c, 5.0);
    mat.insert(r, c, 7.5);

    EXPECT_DOUBLE_EQ(mat.get(r, c), 7.5);
    mat.lock();
    EXPECT_DOUBLE_EQ(mat.get(r, c), 7.5);

    EXPECT_EQ(mat.nnz(), 1u);
    EXPECT_EQ(mat.row_nnz(r), 1u);
}

// ------------------ Bounds checking ------------------

TEST(MatrixBasicTest, BoundsChecks)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(2, 3);

    EXPECT_THROW(mat.insert(2, 0, 1.0), std::out_of_range);
    EXPECT_THROW(mat.insert(0, 3, 1.0), std::out_of_range);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

    EXPECT_THROW(mat.get(2, 0), std::out_of_range);
    EXPECT_THROW(mat.get(0, 3), std::out_of_range);

    EXPECT_THROW(mat.contains(2, 0), std::out_of_range);
    EXPECT_THROW(mat.contains(0, 3), std::out_of_range);

    EXPECT_THROW(mat.row_nnz(2), std::out_of_range);
    EXPECT_NO_THROW(mat.row_nnz(1));

#pragma GCC diagnostic pop
}

// ------------------ clear() ------------------

TEST(MatrixBasicTest, ClearResetsMatrix)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(3, 3);

    mat.insert(0, 0, 1.0);
    mat.insert(1, 1, 2.0);
    mat.insert(2, 2, 3.0);

    EXPECT_FALSE(mat.empty());
    EXPECT_EQ(mat.nnz(), 3u);

    mat.clear();

    EXPECT_TRUE(mat.empty());
    EXPECT_EQ(mat.nnz(), 0u);
    EXPECT_EQ(mat.row_nnz(0), 0u);
    EXPECT_EQ(mat.get(0, 0), 0.0);
    EXPECT_FALSE(mat.contains(1, 1));
}

// ------------------ for_each_row ------------------

TEST(MatrixBasicTest, ForEachRowVisitsAllRowsWithCorrectIndex)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(4, 4);

    mat.insert(0, 1, 1.0);
    mat.insert(2, 3, 2.0);

    std::vector<bool> seen(4, false);

    mat.for_each_row([&](auto const &row, std::size_t idx)
                     {
        ASSERT_LT(idx, 4u);
        seen[idx] = true;

        if (idx == 0) {EXPECT_EQ(row.size(), 1u);}
        if (idx == 1) {EXPECT_EQ(row.size(), 0u);}
        if (idx == 2) {EXPECT_EQ(row.size(), 1u);}
        if (idx == 3) {EXPECT_EQ(row.size(), 0u);} });

    for (bool b : seen)
        EXPECT_TRUE(b);
}

// ------------------ Mode ------------------

TEST(MatrixBasicTest, ModeTransitions)
{
    using I = std::size_t;
    using V = double;
    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(2, 2);

    EXPECT_EQ(mat.mode(), spira::config::matrix_mode::open);
    EXPECT_FALSE(mat.is_locked());

    mat.lock();
    EXPECT_EQ(mat.mode(), spira::config::matrix_mode::locked);
    EXPECT_TRUE(mat.is_locked());

    mat.open();
    EXPECT_EQ(mat.mode(), spira::config::matrix_mode::open);
    EXPECT_FALSE(mat.is_locked());
}
