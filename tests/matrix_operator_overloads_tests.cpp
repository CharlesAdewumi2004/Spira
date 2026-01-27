// tests/test_matrix_ops.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <spira/spira.hpp>


using I = std::size_t;
using V = double;

using LayoutTag = spira::layout::tags::aos_tag;

// -------------------------
// Helpers
// -------------------------
template <class M>
static void expect_same_shape(const M& A, const M& B) {
    EXPECT_EQ(A.get_shape().first,  B.get_shape().first);
    EXPECT_EQ(A.get_shape().second, B.get_shape().second);
}

template <class M>
static void expect_matrix_eq(const M& A, const M& B, double eps = 1e-12) {
    auto [rA, cA] = A.get_shape();
    auto [rB, cB] = B.get_shape();
    ASSERT_EQ(rA, rB);
    ASSERT_EQ(cA, cB);

    for (std::size_t r = 0; r < rA; ++r) {
        for (std::size_t c = 0; c < cA; ++c) {
            V av = A.get(static_cast<I>(r), static_cast<I>(c));
            V bv = B.get(static_cast<I>(r), static_cast<I>(c));
            EXPECT_NEAR(av, bv, eps) << "Mismatch at (" << r << "," << c << ")";
        }
    }
}

template <class M>
static M make_matrix(std::size_t R, std::size_t C,
                     std::initializer_list<std::tuple<I,I,V>> entries)
{
    M A(R, C);

    for (auto [r, c, v] : entries) {
        A.insert(r, c, v);
    }
    return A;
}

// Dense reference ops (slow but correct)
static std::vector<std::vector<V>> dense_add(const std::vector<std::vector<V>>& A,
                                             const std::vector<std::vector<V>>& B)
{
    std::size_t R = A.size(), C = A[0].size();
    std::vector<std::vector<V>> out(R, std::vector<V>(C, 0));
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            out[i][j] = A[i][j] + B[i][j];
    return out;
}

static std::vector<std::vector<V>> dense_sub(const std::vector<std::vector<V>>& A,
                                             const std::vector<std::vector<V>>& B)
{
    std::size_t R = A.size(), C = A[0].size();
    std::vector<std::vector<V>> out(R, std::vector<V>(C, 0));
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            out[i][j] = A[i][j] - B[i][j];
    return out;
}

static std::vector<std::vector<V>> dense_mul(const std::vector<std::vector<V>>& A,
                                             const std::vector<std::vector<V>>& B)
{
    std::size_t R = A.size(), K = A[0].size(), C = B[0].size();
    std::vector<std::vector<V>> out(R, std::vector<V>(C, 0));
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < C; ++j)
                out[i][j] += A[i][k] * B[k][j];
    return out;
}

static std::vector<V> dense_spmv(const std::vector<std::vector<V>>& A,
                                 const std::vector<V>& x)
{
    std::size_t R = A.size(), C = A[0].size();
    std::vector<V> y(R, 0);
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

static std::vector<std::vector<V>> dense_transpose(const std::vector<std::vector<V>>& A)
{
    std::size_t R = A.size(), C = A[0].size();
    std::vector<std::vector<V>> out(C, std::vector<V>(R, 0));
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            out[j][i] = A[i][j];
    return out;
}

template <class M>
static std::vector<std::vector<V>> to_dense(const M& A) {
    auto [R, C] = A.get_shape();
    std::vector<std::vector<V>> out(R, std::vector<V>(C, 0));
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < C; ++c)
            out[r][c] = A.get(static_cast<I>(r), static_cast<I>(c));
    return out;
}

template <class M>
static M from_dense(const std::vector<std::vector<V>>& D) {
    std::size_t R = D.size(), C = D[0].size();
    M A(R, C);
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < C; ++c)
            if (std::abs(D[r][c]) > 0) 
                A.insert(static_cast<I>(r), static_cast<I>(c), D[r][c]);
    return A;
}

// -------------------------
// Tests
// -------------------------
TEST(MatrixOps, AddSubAndInPlace) {
    using M = spira::matrix<LayoutTag, I, V>;

    M A = make_matrix<M>(3, 4, {
        {0,0, 1.0}, {0,3, 2.0},
        {1,1, 3.0},
        {2,2, 4.0}
    });

    M B = make_matrix<M>(3, 4, {
        {0,0, 5.0}, {0,2, 6.0},
        {1,1,-3.0},
        {2,0, 7.0}
    });

    auto dA = to_dense(A);
    auto dB = to_dense(B);

    // +
    M C = A + B;
    M Cref = from_dense<M>(dense_add(dA, dB));
    expect_matrix_eq(C, Cref);

    // -
    M D = A - B;
    M Dref = from_dense<M>(dense_sub(dA, dB));
    expect_matrix_eq(D, Dref);

    // +=
    M A2 = A;
    A2 += B;
    expect_matrix_eq(A2, Cref);

    // -=
    M A3 = A;
    A3 -= B;
    expect_matrix_eq(A3, Dref);
}

TEST(MatrixOps, SpGEMMAndInPlace) {
    using M = spira::matrix<LayoutTag, I, V>;

    // 2x3 * 3x2 = 2x2
    M A = make_matrix<M>(2, 3, {
        {0,0, 1.0}, {0,2, 2.0},
        {1,1, 3.0}
    });

    M B = make_matrix<M>(3, 2, {
        {0,0, 4.0},
        {1,1, 5.0},
        {2,0, 6.0}, {2,1, 7.0}
    });

    auto dA = to_dense(A);
    auto dB = to_dense(B);

    M C = A * B;
    M Cref = from_dense<M>(dense_mul(dA, dB));
    expect_matrix_eq(C, Cref);

    M A2 = A;
    A2 *= B;
    expect_matrix_eq(A2, Cref);
}

TEST(MatrixOps, SpMV) {
    using M = spira::matrix<LayoutTag, I, V>;

    // 3x4
    M A = make_matrix<M>(3, 4, {
        {0,0, 1.0}, {0,1, 2.0},
        {1,2, 3.0},
        {2,3, 4.0}
    });

    std::vector<V> x{10.0, 20.0, 30.0, 40.0};

    auto dA = to_dense(A);
    auto yref = dense_spmv(dA, x);

    auto y = A * x;

    ASSERT_EQ(y.size(), yref.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        EXPECT_NEAR(y[i], yref[i], 1e-12) << "Mismatch at i=" << i;
}

TEST(MatrixOps, ScalarMultiplyDivide) {
    using M = spira::matrix<LayoutTag, I, V>;

    M A = make_matrix<M>(2, 3, {
        {0,0, 1.5}, {0,2, -2.0},
        {1,1, 4.0}
    });

    auto dA = to_dense(A);

    V s = 2.0;
    {
        auto d = dA;
        for (auto& row : d) for (auto& v : row) v *= s;
        M ref = from_dense<M>(d);

        M B = A * s;
        expect_matrix_eq(B, ref);
    }

    {
        auto d = dA;
        for (auto& row : d) for (auto& v : row) v *= s;
        M ref = from_dense<M>(d);

        M B = A;
        B *= s;
        expect_matrix_eq(B, ref);
    }

    {
        auto d = dA;
        for (auto& row : d) for (auto& v : row) v /= s;
        M ref = from_dense<M>(d);

        M B = A / s;
        expect_matrix_eq(B, ref);
    }

    {
        auto d = dA;
        for (auto& row : d) for (auto& v : row) v /= s;
        M ref = from_dense<M>(d);

        M B = A;
        B /= s;
        expect_matrix_eq(B, ref);
    }
}

TEST(MatrixOps, Transpose) {
    using M = spira::matrix<LayoutTag, I, V>;

    M A = make_matrix<M>(2, 3, {
        {0,0, 1.0}, {0,2, 2.0},
        {1,1, 3.0}
    });

    auto dA = to_dense(A);
    auto dT = dense_transpose(dA);
    M Tref = from_dense<M>(dT);

    M T = ~A;

    // shape flip: 2x3 -> 3x2
    EXPECT_EQ(T.get_shape().first,  3u);
    EXPECT_EQ(T.get_shape().second, 2u);

    expect_matrix_eq(T, Tref);
}

TEST(MatrixOps, AddShapeMismatchThrows) {
    using M = spira::matrix<LayoutTag, I, V>;

    M A(2, 2);
    M B(2, 3);

    EXPECT_THROW((void)(A + B), std::invalid_argument);
    EXPECT_THROW((void)(A - B), std::invalid_argument);
}
