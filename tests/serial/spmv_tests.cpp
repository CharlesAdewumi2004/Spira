#include <spira/spira.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <complex>
#include <stdexcept>
#include <vector>

namespace
{

    template <class Mat>
    auto dense_reference_spmv(Mat const &A,
                              std::vector<typename Mat::value_type> const &x)
        -> std::vector<typename Mat::value_type>
    {
        using I = typename Mat::index_type;
        using V = typename Mat::value_type;

        if (x.size() != A.n_cols())
            throw std::invalid_argument("bad x size");

        std::vector<V> y(A.n_rows(), spira::traits::ValueTraits<V>::zero());

        for (std::size_t r = 0; r < A.n_rows(); ++r)
        {
            V acc = spira::traits::ValueTraits<V>::zero();
            for (std::size_t c = 0; c < A.n_cols(); ++c)
            {
                acc += A.get(static_cast<I>(r), static_cast<I>(c)) * x[c];
            }
            y[r] = acc;
        }
        return y;
    }

    template <class V>
    void expect_vec_eq(std::vector<V> const &a, std::vector<V> const &b)
    {
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_EQ(a[i], b[i]);
        }
    }

    inline void expect_vec_eq_double(std::vector<double> const &a, std::vector<double> const &b)
    {
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(a[i], b[i]);
        }
    }

    inline void expect_vec_eq_float(std::vector<float> const &a, std::vector<float> const &b)
    {
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_FLOAT_EQ(a[i], b[i]);
        }
    }

    TEST(SpmvAccuracyTest, IdentityMatrixDouble_AOS)
    {
        using I = std::size_t;
        using V = double;
        spira::matrix<spira::layout::tags::aos_tag, I, V> A(4, 4);

        for (I i = 0; i < 4; ++i)
            A.insert(i, i, 1.0);
        A.lock();

        std::vector<V> x = {1.0, 2.0, 3.0, 4.0};
        std::vector<V> y(4, 0.0);

        spira::serial::algorithms::spmv(A, x, y);

        expect_vec_eq_double(y, x);
    }

    TEST(SpmvAccuracyTest, IdentityMatrixDouble_SOA)
    {
        using I = std::size_t;
        using V = double;
        spira::matrix<spira::layout::tags::soa_tag, I, V> A(4, 4);

        for (I i = 0; i < 4; ++i)
            A.insert(i, i, 1.0);
        A.lock();

        std::vector<V> x = {1.0, 2.0, 3.0, 4.0};
        std::vector<V> y(4, 0.0);

        spira::serial::algorithms::spmv(A, x, y);

        expect_vec_eq_double(y, x);
    }

    TEST(SpmvAccuracyTest, DiagonalMatrixFloat_AOS)
    {
        using I = std::size_t;
        using V = float;
        spira::matrix<spira::layout::tags::aos_tag, I, V> A(4, 4);

        std::vector<V> d = {10.0f, -3.0f, 0.5f, 2.0f};
        for (I i = 0; i < 4; ++i)
            A.insert(i, i, d[i]);
        A.lock();

        std::vector<V> x = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<V> y(4, 0.0f);

        spira::serial::algorithms::spmv(A, x, y);

        std::vector<V> expected = {d[0] * x[0], d[1] * x[1], d[2] * x[2], d[3] * x[3]};
        expect_vec_eq_float(y, expected);
    }

    TEST(SpmvAccuracyTest, DiagonalMatrixFloat_SOA)
    {
        using I = std::size_t;
        using V = float;
        spira::matrix<spira::layout::tags::soa_tag, I, V> A(4, 4);

        std::vector<V> d = {10.0f, -3.0f, 0.5f, 2.0f};
        for (I i = 0; i < 4; ++i)
            A.insert(i, i, d[i]);
        A.lock();

        std::vector<V> x = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<V> y(4, 0.0f);

        spira::serial::algorithms::spmv(A, x, y);

        std::vector<V> expected = {d[0] * x[0], d[1] * x[1], d[2] * x[2], d[3] * x[3]};
        expect_vec_eq_float(y, expected);
    }

    TEST(SpmvAccuracyTest, RectangularOutOfOrderDouble_AOS)
    {
        using I = std::size_t;
        using V = double;

        spira::matrix<spira::layout::tags::aos_tag, I, V> A(3, 5);

        A.insert(0, 0, 1.0);
        A.insert(0, 3, 2.0);
        A.insert(1, 2, 3.0);
        A.insert(2, 4, 5.0);
        A.insert(2, 1, 4.0);

        A.lock();

        std::vector<V> x = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<V> y(3, 0.0);

        spira::serial::algorithms::spmv(A, x, y);

        std::vector<V> expected = {9.0, 9.0, 33.0};
        expect_vec_eq_double(y, expected);
    }

    TEST(SpmvAccuracyTest, RectangularOutOfOrderDouble_SOA)
    {
        using I = std::size_t;
        using V = double;

        spira::matrix<spira::layout::tags::soa_tag, I, V> A(3, 5);

        A.insert(0, 0, 1.0);
        A.insert(0, 3, 2.0);
        A.insert(1, 2, 3.0);
        A.insert(2, 4, 5.0);
        A.insert(2, 1, 4.0);

        A.lock();

        std::vector<V> x = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<V> y(3, 0.0);

        spira::serial::algorithms::spmv(A, x, y);

        std::vector<V> expected = {9.0, 9.0, 33.0};
        expect_vec_eq_double(y, expected);
    }

    TEST(SpmvAccuracyTest, MatchesDenseReference_MixedPattern_AOS)
    {
        using I = std::size_t;
        using V = double;

        spira::matrix<spira::layout::tags::aos_tag, I, V> A(5, 6);

        A.insert(0, 1, 2.0);
        A.insert(0, 4, -1.0);
        A.insert(1, 0, 3.5);
        A.insert(2, 5, 7.0);
        A.insert(3, 2, 1.25);
        A.insert(3, 2, 4.0); // last write wins
        A.insert(4, 3, -2.0);

        A.lock();

        std::vector<V> x = {1, 2, 3, 4, 5, 6};
        std::vector<V> y(A.n_rows(), 0.0);

        auto ref = dense_reference_spmv(A, x);
        spira::serial::algorithms::spmv(A, x, y);
        expect_vec_eq_double(y, ref);
    }

    TEST(SpmvAccuracyTest, MatchesDenseReference_MixedPattern_SOA)
    {
        using I = std::size_t;
        using V = double;

        spira::matrix<spira::layout::tags::soa_tag, I, V> A(5, 6);

        A.insert(0, 1, 2.0);
        A.insert(0, 4, -1.0);
        A.insert(1, 0, 3.5);
        A.insert(2, 5, 7.0);
        A.insert(3, 2, 1.25);
        A.insert(3, 2, 4.0); // last write wins
        A.insert(4, 3, -2.0);

        A.lock();

        std::vector<V> x = {1, 2, 3, 4, 5, 6};
        std::vector<V> y(A.n_rows(), 0.0);

        auto ref = dense_reference_spmv(A, x);
        spira::serial::algorithms::spmv(A, x, y);
        expect_vec_eq_double(y, ref);
    }

    TEST(SpmvAccuracyTest, EmptyMatrixProducesZeroVector_AOS)
    {
        using I = std::size_t;
        using V = double;
        spira::matrix<spira::layout::tags::aos_tag, I, V> A(4, 5);
        A.lock();

        std::vector<V> x(5, 3.14);
        std::vector<V> y(4, 123.0);

        spira::serial::algorithms::spmv(A, x, y);

        for (auto v : y)
            EXPECT_DOUBLE_EQ(v, 0.0);
    }

    TEST(SpmvAccuracyTest, EmptyMatrixProducesZeroVector_SOA)
    {
        using I = std::size_t;
        using V = double;
        spira::matrix<spira::layout::tags::soa_tag, I, V> A(4, 5);
        A.lock();

        std::vector<V> x(5, 3.14);
        std::vector<V> y(4, 123.0);

        spira::serial::algorithms::spmv(A, x, y);

        for (auto v : y)
            EXPECT_DOUBLE_EQ(v, 0.0);
    }

    TEST(SpmvAccuracyTest, DimensionMismatchThrows_AOS)
    {
        using I = std::size_t;
        using V = double;
        spira::matrix<spira::layout::tags::aos_tag, I, V> A(3, 4);

        std::vector<V> x_bad(3, 1.0); // should be 4
        std::vector<V> y(3, 0.0);
        EXPECT_THROW(spira::serial::algorithms::spmv(A, x_bad, y), std::invalid_argument);

        std::vector<V> x(4, 1.0);
        std::vector<V> y_bad(2, 0.0); // should be 3
        EXPECT_THROW(spira::serial::algorithms::spmv(A, x, y_bad), std::invalid_argument);
    }

    TEST(SpmvAccuracyTest, DimensionMismatchThrows_SOA)
    {
        using I = std::size_t;
        using V = double;
        spira::matrix<spira::layout::tags::soa_tag, I, V> A(3, 4);

        std::vector<V> x_bad(3, 1.0);
        std::vector<V> y(3, 0.0);
        EXPECT_THROW(spira::serial::algorithms::spmv(A, x_bad, y), std::invalid_argument);

        std::vector<V> x(4, 1.0);
        std::vector<V> y_bad(2, 0.0);
        EXPECT_THROW(spira::serial::algorithms::spmv(A, x, y_bad), std::invalid_argument);
    }

    TEST(SpmvAccuracyTest, ComplexMatrixMultiply_AOS)
    {
        using I = std::size_t;
        using cd = std::complex<double>;

        spira::matrix<spira::layout::tags::aos_tag, I, cd> A(2, 2);

        A.insert(0, 0, cd(1.0, 2.0));
        A.insert(0, 1, cd(3.0, 0.0));
        A.insert(1, 0, cd(0.0, 1.0));
        A.insert(1, 1, cd(2.0, -1.0));
        A.lock();

        std::vector<cd> x = {cd(1.0, -1.0), cd(2.0, 2.0)};
        std::vector<cd> y(2, cd(0.0, 0.0));

        spira::serial::algorithms::spmv(A, x, y);

        std::vector<cd> expected = {cd(9.0, 7.0), cd(7.0, 3.0)};
        expect_vec_eq(y, expected);
    }

    TEST(SpmvAccuracyTest, ComplexMatrixMultiply_SOA)
    {
        using I = std::size_t;
        using cd = std::complex<double>;

        spira::matrix<spira::layout::tags::soa_tag, I, cd> A(2, 2);

        A.insert(0, 0, cd(1.0, 2.0));
        A.insert(0, 1, cd(3.0, 0.0));
        A.insert(1, 0, cd(0.0, 1.0));
        A.insert(1, 1, cd(2.0, -1.0));
        A.lock();

        std::vector<cd> x = {cd(1.0, -1.0), cd(2.0, 2.0)};
        std::vector<cd> y(2, cd(0.0, 0.0));

        spira::serial::algorithms::spmv(A, x, y);

        std::vector<cd> expected = {cd(9.0, 7.0), cd(7.0, 3.0)};
        expect_vec_eq(y, expected);
    }

}
