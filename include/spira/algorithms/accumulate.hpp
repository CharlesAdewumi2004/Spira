#pragma once
#include "spira/concepts.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{
    /// Accumulate (sum) a single row. Works in both open and locked mode.
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    V accumulate(spira::matrix<Layout, I, V> const &mat, size_t i){
        if (i >= mat.shape().first) {
            throw std::out_of_range("Row does not exist in matrix");
        }

        return mat.accumulate(static_cast<I>(i));
    }

    /// Accumulate (sum) every row. Works in both open and locked mode.
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    std::vector<V> accumulate(spira::matrix<Layout, I, V> const &mat){
        size_t num_of_rows = mat.shape().first;
        std::vector<V> acc(num_of_rows);

        for(size_t i = 0; i < num_of_rows; i++){
            acc[i] = mat.accumulate(static_cast<I>(i));
        }

        return acc;
    }

} // namespace spira::algorithms
