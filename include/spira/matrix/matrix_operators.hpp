#pragma once

#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/algorithms/merge.hpp>
#include <spira/algorithms/spgemm.hpp>
#include <spira/algorithms/spmv.hpp>
#include <spira/algorithms/transpose.hpp>
#include <spira/algorithms/matrix_addition.hpp>
#include <spira/algorithms/scalars.hpp>

namespace spira
{

    // =====================
    // Add / Sub
    // =====================

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V>
    matrix<LayoutTag, I, V>::operator+(const matrix &other) const
    {
        return algorithms::MatrixAddition(*this, other);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V>
    matrix<LayoutTag, I, V>::operator-(const matrix &other) const
    {
        matrix out(other);
        algorithms::multiplication_scaler(other, out, V{-1});
        return algorithms::MatrixAddition(*this, out);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V> &
    matrix<LayoutTag, I, V>::operator+=(const matrix &other)
    {
        *this = algorithms::MatrixAddition(*this, other);
        return *this;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V> &
    matrix<LayoutTag, I, V>::operator-=(const matrix &other)
    {
        matrix out(other);
        algorithms::multiplication_scaler(other, out, V{-1});
        *this = algorithms::MatrixAddition(*this, out);
        return *this;
    }

    // =====================
    // SpGEMM
    // =====================

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V>
    matrix<LayoutTag, I, V>::operator*(const matrix &other) const
    {
        return algorithms::spgemm(*this, other);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V> &
    matrix<LayoutTag, I, V>::operator*=(const matrix &other)
    {
        *this = algorithms::spgemm(*this, other);
        return *this;
    }

    // =====================
    // SpMV
    // =====================

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline std::vector<V>
    matrix<LayoutTag, I, V>::operator*(const std::vector<V> &x) const
    {
        std::vector<V> y(this->n_rows());
        algorithms::spmv(*this, x, y);
        return y;
    }

    // =====================
    // Scalar ops
    // =====================

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V>
    matrix<LayoutTag, I, V>::operator*(V s) const
    {
        matrix out(*this);
        algorithms::multiplication_scaler(*this, out, s);
        return out;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V> &
    matrix<LayoutTag, I, V>::operator*=(V s)
    {
        algorithms::multiplication_scaler(*this, s);
        return *this;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V>
    matrix<LayoutTag, I, V>::operator/(V s) const
    {
        matrix out(*this);
        algorithms::division_scaler(*this, out, s);
        return out;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V> &
    matrix<LayoutTag, I, V>::operator/=(V s)
    {
        algorithms::division_scaler(*this, s);
        return *this;
    }

    // =====================
    // Transpose
    // =====================

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    inline matrix<LayoutTag, I, V>
    matrix<LayoutTag, I, V>::operator~() const
    {
        return algorithms::transpose(*this);
    }

}
