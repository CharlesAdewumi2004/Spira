#include <spira/spira.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace {

using I = std::size_t;
using V = double;
using key = std::pair<I, I>;

constexpr I N = 10000;
constexpr int INSERTS = 100000;

// Ground truth for a random insert stream: last write wins, zero erases.
struct expected_entries
{
    std::map<key, std::pair<V, std::size_t>> map; // value, position in live
    std::vector<key> live;                        // for O(1) random picks

    void write(const key &k, V val)
    {
        auto it = map.find(k);
        if (val == 0.0)
        {
            if (it == map.end())
                return;
            const std::size_t pos = it->second.second;
            live[pos] = live.back();
            map[live[pos]].second = pos;
            live.pop_back();
            map.erase(k);
        }
        else if (it != map.end())
        {
            it->second.first = val;
        }
        else
        {
            map.emplace(k, std::pair{val, live.size()});
            live.push_back(k);
        }
    }
};

// 100k inserts into an N x N matrix: the first 10k at random positions, then
// alternating overwrites of existing entries and fresh random positions, with
// values in [-10, 10] so about 1 in 21 writes is a deletion.
template <class LayoutTag>
expected_entries fill_randomly(spira::matrix<LayoutTag, I, V> &mat, std::mt19937_64 &rng)
{
    std::uniform_int_distribution<I> dist_index(0, N - 1);
    std::uniform_int_distribution<int> dist_val(-10, 10);
    expected_entries expected;

    for (int k = 0; k < INSERTS; ++k)
    {
        key pos{dist_index(rng), dist_index(rng)};
        if (k >= 10000 && k % 2 == 0 && !expected.live.empty())
            pos = expected.live[dist_index(rng) % expected.live.size()];

        const V val = static_cast<V>(dist_val(rng));
        expected.write(pos, val);
        mat.insert(pos.first, pos.second, val);
    }
    return expected;
}

template <class LayoutTag>
void bulk_insert_lock_integrity_and_spmv()
{
    spira::matrix<LayoutTag, I, V> mat(N, N);
    std::mt19937_64 rng(42);
    const expected_entries expected = fill_randomly(mat, rng);
    mat.lock();

    ASSERT_EQ(mat.nnz(), expected.map.size());

    for (const auto &[k, entry] : expected.map)
    {
        EXPECT_TRUE(mat.contains(k.first, k.second));
        EXPECT_DOUBLE_EQ(mat.get(k.first, k.second), entry.first);
    }

    std::uniform_int_distribution<I> dist_index(0, N - 1);
    for (int t = 0; t < 200; ++t)
    {
        const key k{dist_index(rng), dist_index(rng)};
        if (!expected.map.contains(k))
        {
            EXPECT_DOUBLE_EQ(mat.get(k.first, k.second), 0.0);
        }
    }

    std::uniform_real_distribution<double> dist_x(-1.0, 1.0);
    std::vector<V> x(N);
    for (auto &xi : x)
        xi = dist_x(rng);

    std::vector<V> y_expected(N, 0.0);
    for (const auto &[k, entry] : expected.map)
        y_expected[k.first] += entry.first * x[k.second];

    std::vector<V> y(N, 0.0);
    spira::serial::algorithms::spmv(mat, x, y);

    for (I i = 0; i < N; ++i)
    {
        const double tol = 1e-10 + 1e-10 * std::max(std::abs(y[i]), std::abs(y_expected[i]));
        EXPECT_LE(std::abs(y[i] - y_expected[i]), tol) << "row " << i;
    }
}

} // namespace

TEST(LargeMatrixStressTest, BulkInsertLockIntegrityAndSpmv_AOS)
{
    bulk_insert_lock_integrity_and_spmv<spira::layout::tags::aos_tag>();
}

TEST(LargeMatrixStressTest, BulkInsertLockIntegrityAndSpmv_SOA)
{
    bulk_insert_lock_integrity_and_spmv<spira::layout::tags::soa_tag>();
}
