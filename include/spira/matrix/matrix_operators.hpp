#pragma once

#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/serial/spgemm.hpp>
#include <spira/serial/spmv.hpp>
#include <spira/serial/transpose.hpp>
#include <spira/serial/matrix_addition.hpp>
#include <spira/serial/scalars.hpp>

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
        return serial::algorithms::MatrixAddition(*this, other);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator-(const matrix &other) const
    {
        if (this->shape() != other.shape())
            throw std::invalid_argument("operator-: matrix shapes must match");
        matrix<L, I, V, BT, BN, LP> out(other.shape().first, other.shape().second);
        serial::algorithms::multiplication_scaler(other, out, V{-1});
        return serial::algorithms::MatrixAddition(*this, out);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator+=(const matrix &other)
    {
        *this = serial::algorithms::MatrixAddition(*this, other);
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
        serial::algorithms::multiplication_scaler(other, out, V{-1});
        *this = serial::algorithms::MatrixAddition(*this, out);
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
        return serial::algorithms::spgemm(*this, other);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator*=(const matrix &other)
    {
        *this = serial::algorithms::spgemm(*this, other);
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
        serial::algorithms::spmv(*this, x, y);
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
        serial::algorithms::multiplication_scaler(*this, out, s);
        return out;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator*=(V s)
    {
        serial::algorithms::multiplication_scaler(*this, s);
        return *this;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP>
    matrix<L, I, V, BT, BN, LP>::operator/(V s) const
    {
        matrix out(*this);
        serial::algorithms::division_scaler(*this, out, s);
        return out;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    inline matrix<L, I, V, BT, BN, LP> &
    matrix<L, I, V, BT, BN, LP>::operator/=(V s)
    {
        serial::algorithms::division_scaler(*this, s);
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
        return serial::algorithms::transpose(*this);
    }

} // namespace spira
