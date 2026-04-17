// tests/parallel/partition_tests.cpp
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <spira/parallel/partition.hpp>

using namespace spira::parallel;

// ─────────────────────────────────────────────────────────────────────────────
// compute_partition_boundaries
// ─────────────────────────────────────────────────────────────────────────────

// Structural invariants that must hold for every valid output.
TEST(PartitionBoundaries, OutputContract)
{
    const std::vector<std::size_t> nnz = {10, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    for (std::size_t n : {1u, 2u, 4u, 6u, 10u})
    {
        const auto b = compute_partition_boundaries(nnz, n);

        ASSERT_EQ(b.size(), n + 1) << "n=" << n;
        EXPECT_EQ(b.front(), 0u)   << "n=" << n;
        EXPECT_EQ(b.back(), 10u)   << "n=" << n;

        for (std::size_t i = 1; i < b.size(); ++i)
            EXPECT_LE(b[i - 1], b[i]) << "non-monotonic at i=" << i << " n=" << n;
    }
}

// Uniform nnz should split rows into equal-sized contiguous ranges.
TEST(PartitionBoundaries, UniformDensity)
{
    // 12 rows × 4 nnz, 3 threads → [0, 4, 8, 12]
    {
        const auto b = compute_partition_boundaries(std::vector<std::size_t>(12, 4), 3);
        EXPECT_EQ(b[1], 4u);
        EXPECT_EQ(b[2], 8u);
    }
    // 10 rows × 5 nnz, 2 threads → [0, 5, 10]
    {
        const auto b = compute_partition_boundaries(std::vector<std::size_t>(10, 5), 2);
        EXPECT_EQ(b[1], 5u);
    }
}

// Each partition should receive approximately total_nnz / n_threads entries.
TEST(PartitionBoundaries, BalancedLoad)
{
    constexpr std::size_t n_rows    = 100;
    constexpr std::size_t n_threads = 7;
    const std::vector<std::size_t> nnz(n_rows, 3);
    const auto b = compute_partition_boundaries(nnz, n_threads);

    const std::size_t target = (n_rows * 3) / n_threads;

    for (std::size_t t = 0; t < n_threads; ++t)
    {
        std::size_t part_nnz = 0;
        for (std::size_t r = b[t]; r < b[t + 1]; ++r)
            part_nnz += nnz[r];

        EXPECT_LE(part_nnz, target + 3) << "thread " << t;
        if (b[t + 1] > b[t])
        {
            EXPECT_GE(part_nnz, target - 3) << "thread " << t;
        }
    }
}

// A dominant row must land in its own small partition; heavier rows mean fewer
// rows per partition.
TEST(PartitionBoundaries, SkewedDensity)
{
    // Row 0 carries > half the total nnz → thread 0 must get just row 0.
    {
        std::vector<std::size_t> nnz(10, 1);
        nnz[0] = 1000;
        const auto b = compute_partition_boundaries(nnz, 2);
        EXPECT_EQ(b[1], 1u) << "heavy row 0 should be alone in partition 0";
    }
    // First 4 rows heavy → thread 0 gets fewer rows than thread 1.
    {
        std::vector<std::size_t> nnz(10, 1);
        for (int i = 0; i < 4; ++i)
            nnz[i] = 100;
        const auto b = compute_partition_boundaries(nnz, 2);
        EXPECT_LT(b[1] - b[0], b[2] - b[1]);
    }
}

// When n_threads > n_rows, excess threads must receive empty partitions and
// every row must be owned by exactly one thread.
TEST(PartitionBoundaries, MoreThreadsThanRows)
{
    const std::vector<std::size_t> nnz = {5, 5, 5};
    const auto b = compute_partition_boundaries(nnz, 10);

    // Single-row case: first thread gets the only row, rest are empty.
    {
        const auto b1 = compute_partition_boundaries({42}, 8);
        EXPECT_EQ(b1[1], 1u);
        for (std::size_t i = 2; i <= 8; ++i)
            EXPECT_EQ(b1[i], 1u) << "boundary[" << i << "] should mark empty partition";
    }

    // Coverage: each row owned exactly once.
    std::vector<int> covered(3, 0);
    for (std::size_t t = 0; t < 10; ++t)
        for (std::size_t r = b[t]; r < b[t + 1]; ++r)
            covered[r]++;

    for (std::size_t r = 0; r < 3; ++r)
        EXPECT_EQ(covered[r], 1) << "row " << r;
}

// Degenerate inputs: empty row vector and single thread.
TEST(PartitionBoundaries, EdgeCases)
{
    // Empty matrix → all boundaries are 0.
    {
        const auto b = compute_partition_boundaries({}, 4);
        ASSERT_EQ(b.size(), 5u);
        for (auto v : b)
            EXPECT_EQ(v, 0u);
    }
    // Single thread → one partition covering all rows.
    {
        const auto b = compute_partition_boundaries({1, 2, 3, 4, 5}, 1);
        ASSERT_EQ(b.size(), 2u);
        EXPECT_EQ(b[0], 0u);
        EXPECT_EQ(b[1], 5u);
    }
}

// All-zero nnz must fall back to a uniform row-count split.
TEST(PartitionBoundaries, AllZeroNnz)
{
    const auto b = compute_partition_boundaries(std::vector<std::size_t>(9, 0), 3);
    EXPECT_EQ(b[1], 3u);
    EXPECT_EQ(b[2], 6u);
    EXPECT_EQ(b[3], 9u);
}

// ─────────────────────────────────────────────────────────────────────────────
// partition struct
// ─────────────────────────────────────────────────────────────────────────────

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
