#pragma once

#include <stdexcept>
#include <type_traits>
#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/serial/matrix_addition.hpp>
#include <spira/serial/scalars.hpp>
#include <spira/serial/spgemm.hpp>
#include <spira/serial/spmv.hpp>
#include <spira/serial/transpose.hpp>

// Arithmetic on locked matrices. Each operator returns (or leaves) a locked
// matrix built by the matching serial algorithm; the compound forms rebuild the
// left operand.

namespace spira
{

    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> operator+(const matrix<L, I, V, BT, BN, S> &a, const matrix<L, I, V, BT, BN, S> &b)
    {
        return serial::algorithms::MatrixAddition(a, b);
    }

    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> operator*(const matrix<L, I, V, BT, BN, S> &m, std::type_identity_t<V> s)
    {
        matrix<L, I, V, BT, BN, S> out(m.n_rows(), m.n_cols());
        serial::algorithms::multiplication_scaler(m, out, s);
        return out;
    }

    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> operator/(const matrix<L, I, V, BT, BN, S> &m, std::type_identity_t<V> s)
    {
        matrix<L, I, V, BT, BN, S> out(m.n_rows(), m.n_cols());
        serial::algorithms::division_scaler(m, out, s);
        return out;
    }

    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> operator-(const matrix<L, I, V, BT, BN, S> &a, const matrix<L, I, V, BT, BN, S> &b)
    {
        if (a.shape() != b.shape())
            throw std::invalid_argument("operator-: matrix shapes must match");
        return a + b * V{-1};
    }

    /// SpGEMM.
    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> operator*(const matrix<L, I, V, BT, BN, S> &a, const matrix<L, I, V, BT, BN, S> &b)
    {
        return serial::algorithms::spgemm(a, b);
    }

    /// SpMV: returns a · x.
    template <class L, class I, class V, class BT, std::size_t BN, class S>
    std::vector<V> operator*(const matrix<L, I, V, BT, BN, S> &a, const std::vector<V> &x)
    {
        std::vector<V> y(a.n_rows());
        serial::algorithms::spmv(a, x, y);
        return y;
    }

    /// Transpose.
    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> operator~(const matrix<L, I, V, BT, BN, S> &a)
    {
        return serial::algorithms::transpose(a);
    }

    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> &operator+=(matrix<L, I, V, BT, BN, S> &a, const matrix<L, I, V, BT, BN, S> &b)
    {
        return a = a + b;
    }

    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> &operator-=(matrix<L, I, V, BT, BN, S> &a, const matrix<L, I, V, BT, BN, S> &b)
    {
        return a = a - b;
    }

    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> &operator*=(matrix<L, I, V, BT, BN, S> &a, const matrix<L, I, V, BT, BN, S> &b)
    {
        return a = a * b;
    }

    /// In place: a must be open, and stays open (see multiplication_scaler).
    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> &operator*=(matrix<L, I, V, BT, BN, S> &a, std::type_identity_t<V> s)
    {
        serial::algorithms::multiplication_scaler(a, s);
        return a;
    }

    /// In place: a must be open, and stays open (see division_scaler).
    template <class L, class I, class V, class BT, std::size_t BN, class S>
    matrix<L, I, V, BT, BN, S> &operator/=(matrix<L, I, V, BT, BN, S> &a, std::type_identity_t<V> s)
    {
        serial::algorithms::division_scaler(a, s);
        return a;
    }

} // namespace spira
