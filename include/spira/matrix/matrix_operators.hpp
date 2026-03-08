#pragma once

#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
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

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator+(const matrix &other) const
    {
        return algorithms::MatrixAddition(*this, other);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator-(const matrix &other) const
    {
        if (this->shape() != other.shape())
            throw std::invalid_argument("operator-: matrix shapes must match");
        matrix<L, I, V, BT, BN, LP> out(other.shape().first, other.shape().second);
        algorithms::multiplication_scaler(other, out, V{-1});
        return algorithms::MatrixAddition(*this, out);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator+=(const matrix &other)
    {
        *this = algorithms::MatrixAddition(*this, other);
        return *this;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator-=(const matrix &other)
    {
        if (this->shape() != other.shape())
            throw std::invalid_argument("operator-=: matrix shapes must match");
        matrix<L, I, V, BT, BN, LP> out(other.shape().first, other.shape().second);
        algorithms::multiplication_scaler(other, out, V{-1});
        *this = algorithms::MatrixAddition(*this, out);
        return *this;
    }

    // =====================
    // SpGEMM
    // =====================

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator*(const matrix &other) const
    {
        return algorithms::spgemm(*this, other);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator*=(const matrix &other)
    {
        *this = algorithms::spgemm(*this, other);
        return *this;
    }

    // =====================
    // SpMV
    // =====================

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline std::vector<V>
    matrix<L, I, V, BT, BN, LP>::operator*(const std::vector<V> &x) const
    {
        std::vector<V> y(this->n_rows());
        algorithms::spmv(*this, x, y);
        return y;
    }

    // =====================
    // Scalar ops
    // =====================

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator*(V s) const
    {
        matrix out(*this);
        algorithms::multiplication_scaler(*this, out, s);
        return out;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator*=(V s)
    {
        algorithms::multiplication_scaler(*this, s);
        return *this;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator/(V s) const
    {
        matrix out(*this);
        algorithms::division_scaler(*this, out, s);
        return out;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator/=(V s)
    {
        algorithms::division_scaler(*this, s);
        return *this;
    }

    // =====================
    // Transpose
    // =====================

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator~() const
    {
        return algorithms::transpose(*this);
    }

} // namespace spira
