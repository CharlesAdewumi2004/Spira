#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <spira/matrix/matrix.hpp>

namespace spira::serial::algorithms
{
    /// Accumulate (sum) a single row. Works in both open and locked mode.
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V, class BT, std::size_t BN, class S>
    V accumulate(const spira::matrix<Layout, I, V, BT, BN, S> &mat, size_t i){
        if (i >= mat.shape().first) {
            throw std::out_of_range("Row does not exist in matrix");
        }

        return mat.accumulate(static_cast<I>(i));
    }

    /// Accumulate (sum) every row. Works in both open and locked mode.
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V, class BT, std::size_t BN, class S>
    std::vector<V> accumulate(const spira::matrix<Layout, I, V, BT, BN, S> &mat){
        size_t num_of_rows = mat.shape().first;
        std::vector<V> acc(num_of_rows);

        for(size_t i = 0; i < num_of_rows; i++){
            acc[i] = mat.accumulate(static_cast<I>(i));
        }

        return acc;
    }

} // namespace spira::serial::algorithms
