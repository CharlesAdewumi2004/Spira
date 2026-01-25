#include <spira/spira.hpp>
#include <gtest/gtest.h>
#include <cstddef>

// ------------------ ManyInsertsThenOverwrite_DedupAfterFlush ------------------

TEST(FlushBehaviorTest, ManyInsertsThenOverwrite_DedupAfterFlush_AOS) {
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(1, 10);
    mat.set_mode(spira::mode::matrix_mode::spmv);

    mat.insert(0, 1, 1.0);
    mat.insert(0, 2, 2.0);
    mat.insert(0, 3, 3.0);
    mat.insert(0, 4, 4.0);

    mat.insert(0, 1, 5.0);

    EXPECT_TRUE(mat.contains(0, 1));
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 5.0);

    mat.flush();

    EXPECT_DOUBLE_EQ(mat.get(0, 1), 5.0);
    EXPECT_TRUE(mat.contains(0, 2));
    EXPECT_TRUE(mat.contains(0, 3));
    EXPECT_TRUE(mat.contains(0, 4));

    EXPECT_EQ(mat.row_nnz(0), 4u);
    EXPECT_EQ(mat.nnz(), 4u);
}

TEST(FlushBehaviorTest, ManyInsertsThenOverwrite_DedupAfterFlush_SOA) {
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(1, 10);
    mat.set_mode(spira::mode::matrix_mode::spmv);

    mat.insert(0, 1, 1.0);
    mat.insert(0, 2, 2.0);
    mat.insert(0, 3, 3.0);
    mat.insert(0, 4, 4.0);

    mat.insert(0, 1, 5.0);

    EXPECT_TRUE(mat.contains(0, 1));
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 5.0);

    mat.flush();

    EXPECT_DOUBLE_EQ(mat.get(0, 1), 5.0);
    EXPECT_TRUE(mat.contains(0, 2));
    EXPECT_TRUE(mat.contains(0, 3));
    EXPECT_TRUE(mat.contains(0, 4));

    EXPECT_EQ(mat.row_nnz(0), 4u);
    EXPECT_EQ(mat.nnz(), 4u);
}


// ------------------ FlushIsIdempotentAndPreservesLogicalState ------------------

TEST(FlushBehaviorTest, FlushIsIdempotentAndPreservesLogicalState_AOS) {
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::aos_tag, I, V> mat(2, 2);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    mat.insert(0, 0, 7.7);
    mat.insert(0, 1, 8.8);

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_TRUE(mat.contains(0, 0));
    EXPECT_TRUE(mat.contains(0, 1));
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 7.7);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 8.8);

    mat.flush();

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 8.8);

    // overwrite then flush again
    mat.insert(0, 1, 9.9);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 9.9);

    mat.flush();

    EXPECT_DOUBLE_EQ(mat.get(0, 1), 9.9);
    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}

TEST(FlushBehaviorTest, FlushIsIdempotentAndPreservesLogicalState_SOA) {
    using I = std::size_t;
    using V = double;

    spira::matrix<spira::layout::tags::soa_tag, I, V> mat(2, 2);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    mat.insert(0, 0, 7.7);
    mat.insert(0, 1, 8.8);

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_TRUE(mat.contains(0, 0));
    EXPECT_TRUE(mat.contains(0, 1));
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 7.7);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 8.8);

    mat.flush();

    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 8.8);

    mat.insert(0, 1, 9.9);
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 9.9);

    mat.flush();

    EXPECT_DOUBLE_EQ(mat.get(0, 1), 9.9);
    EXPECT_EQ(mat.row_nnz(0), 2u);
    EXPECT_EQ(mat.nnz(), 2u);
}
