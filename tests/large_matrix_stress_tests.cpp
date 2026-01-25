#include <spira/spira.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <map>
#include <random>
#include <utility>

namespace {

template<class LayoutTag>
void bulk_insert_mode_switch_spmv_correctness()
{
    using I = std::size_t;
    using V = double;

    constexpr I N = 10000;
    constexpr int INSERTS = 100000;

    spira::matrix<LayoutTag, I, V> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<I> dist_index(0, N - 1);
    std::uniform_int_distribution<int> dist_val(-10, 10);

    // ground truth sparse storage (dedupbed; zero == erased)
    std::map<std::pair<I, I>, V> expected;

    for (int k = 0; k < INSERTS; ++k) {
        I i, j;

        if (k < 10000) {
            i = dist_index(rng);
            j = dist_index(rng);
        } else {
            if (!expected.empty() && (k % 2 == 0)) {
                auto it = expected.begin();
                std::advance(it, static_cast<std::ptrdiff_t>(dist_index(rng) % expected.size()));
                i = it->first.first;
                j = it->first.second;
            } else {
                i = dist_index(rng);
                j = dist_index(rng);
            }
        }

        V val = static_cast<V>(dist_val(rng));
        if (val == 0.0) {
            expected.erase({i, j});
        } else {
            expected[{i, j}] = val;
        }

        mat.insert(i, j, val);
    }

    // force materialization
    mat.flush();

    // switch to spmv mode and ensure internal state is consistent
    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    ASSERT_EQ(mat.nnz(), expected.size());

    // ----- Build random x -----
    std::uniform_real_distribution<double> dist_x(-1.0, 1.0);
    std::vector<V> x(N);
    for (I j = 0; j < N; ++j) x[j] = dist_x(rng);

    // ----- Reference y = A*x from expected map -----
    std::vector<V> y_expected(N, 0.0);
    for (auto const& [key, aij] : expected) {
        const I i = key.first;
        const I j = key.second;
        y_expected[i] += aij * x[j];
    }

    // ----- Spira spmv -----
    std::vector<V> y(N, 0.0);
    spira::algorithms::spmv(mat, x, y);

    // ----- Compare -----
    // Values are small-ish; still use a tolerance because order of accumulation can differ.
    constexpr double ABS_EPS = 1e-10;
    constexpr double REL_EPS = 1e-10;

    for (I i = 0; i < N; ++i) {
        const double a = y[i];
        const double b = y_expected[i];
        const double diff = std::abs(a - b);
        const double tol = ABS_EPS + REL_EPS * std::max(std::abs(a), std::abs(b));
        EXPECT_LE(diff, tol) << "row i=" << i << " y=" << a << " expected=" << b;
    }
}



TEST(LargeMatrixSpmvTest, BulkInsertModeSwitchSpmvCorrectness_AOS) {
    bulk_insert_mode_switch_spmv_correctness<spira::layout::tags::aos_tag>();
}

TEST(LargeMatrixSpmvTest, BulkInsertModeSwitchSpmvCorrectness_SOA) {
    bulk_insert_mode_switch_spmv_correctness<spira::layout::tags::soa_tag>();
}

template<class LayoutTag>
void bulk_insert_and_mode_switch_integrity()
{
    using I = std::size_t;
    using V = double;

    constexpr I N = 10000;
    constexpr int INSERTS = 100000; 

    spira::matrix<LayoutTag, I, V> mat(N, N);
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<I> dist_index(0, N - 1);
    std::uniform_int_distribution<int> dist_val(-10, 10);

    std::map<std::pair<I, I>, V> expected;

    for (int k = 0; k < INSERTS; ++k) {
        I i, j;

        if (k < 10000) {
            i = dist_index(rng);
            j = dist_index(rng);
        } else {
            if (!expected.empty() && (k % 2 == 0)) {
                auto it = expected.begin();
                std::advance(it, static_cast<std::ptrdiff_t>(dist_index(rng) % expected.size()));
                i = it->first.first;
                j = it->first.second;
            } else {
                i = dist_index(rng);
                j = dist_index(rng);
            }
        }

        V val = static_cast<V>(dist_val(rng));
        if (val == 0.0) {
            expected.erase({i, j});
        } else {
            expected[{i, j}] = val;
        }

        mat.insert(i, j, val);
    }

    mat.flush();

    EXPECT_EQ(mat.nnz(), expected.size());

    for (auto const& [key, vexp] : expected) {
        I i = key.first;
        I j = key.second;
        EXPECT_TRUE(mat.contains(i, j));
        EXPECT_DOUBLE_EQ(mat.get(i, j), vexp);
    }

    for (int t = 0; t < 200; ++t) {
        I i = dist_index(rng);
        I j = dist_index(rng);
        if (expected.find({i, j}) == expected.end()) {
            EXPECT_DOUBLE_EQ(mat.get(i, j), 0.0);
        }
    }

    mat.set_mode(spira::mode::matrix_mode::spmv);
    mat.flush();

    EXPECT_EQ(mat.nnz(), expected.size());
    for (auto const& [key, vexp] : expected) {
        I i = key.first;
        I j = key.second;
        EXPECT_TRUE(mat.contains(i, j));
        EXPECT_DOUBLE_EQ(mat.get(i, j), vexp);
    }
}

}


TEST(LargeMatrixStressTest, BulkInsertAndModeSwitchIntegrity_AOS) {
    bulk_insert_and_mode_switch_integrity<spira::layout::tags::aos_tag>();
}

TEST(LargeMatrixStressTest, BulkInsertAndModeSwitchIntegrity_SOA) {
    bulk_insert_and_mode_switch_integrity<spira::layout::tags::soa_tag>();
}
