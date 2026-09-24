// tests/parallel/partition_tests.cpp
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <spira/parallel/partition.hpp>

using namespace spira::parallel;

using Layout = spira::layout::tags::aos_tag;
using PartT  = partition<Layout, uint32_t, double>;

// size() and local_row() arithmetic.
TEST(PartitionStruct, SizeAndLocalRow)
{
    PartT p;

    p.row_start = 0; p.row_end = 0;
    EXPECT_EQ(p.size(), 0u);

    p.row_start = 10; p.row_end = 25;
    EXPECT_EQ(p.size(), 15u);
    EXPECT_EQ(p.local_row(10), 0u);
    EXPECT_EQ(p.local_row(17), 7u);
    EXPECT_EQ(p.local_row(24), 14u);
}

// Default state and compile-time check that SoA layout instantiates correctly.
TEST(PartitionStruct, DefaultAndLayouts)
{
    // Default-constructed is empty.
    PartT p;
    EXPECT_EQ(p.row_start, 0u);
    EXPECT_EQ(p.row_end,   0u);
    EXPECT_TRUE(p.rows.empty());

    // SoA variant compiles and behaves identically.
    partition<spira::layout::tags::soa_tag, uint32_t, float> soa;
    soa.row_start = 3;
    soa.row_end   = 7;
    EXPECT_EQ(soa.size(), 4u);
    EXPECT_EQ(soa.local_row(5), 2u);
}
