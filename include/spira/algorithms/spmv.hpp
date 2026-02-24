#pragma once

#include "spira/matrix/layouts/layout_tags.hpp"
#include <cstdint>
#include <spira/kernels/kernels.h>
#include <spira/matrix/matrix.hpp>
#include <type_traits>
#include <vector>

namespace spira::algorithms {
template <class Layout, concepts::Indexable I, concepts::Valueable V>
inline void spmv(const spira::matrix<Layout, I, V> &matrix, const std::vector<V> &x, std::vector<V> &y) {
    if (x.size() != matrix.n_cols()) {
        throw std::invalid_argument(
            "The size of the input vector x does not match the number of columns of the matrix");
    }
    if (y.size() != matrix.n_rows()) {
        throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
    }

    auto SpMV = [&y, &x](const row<Layout, I, V> &row, I rowIndex) {
        V acc = traits::ValueTraits<V>::zero();

        auto op = [&acc, &x](I const &col, V const &val) { acc += x[col] * val; };

        row.for_each_element(op);

        y[rowIndex] = acc;
    };

    matrix.for_each_row(SpMV);
}

template <>
inline void
spmv<layout::tags::soa_tag, uint32_t, float>(const spira::matrix<layout::tags::soa_tag, uint32_t, float> &matrix,
                                             const std::vector<float> &x, std::vector<float> &y) {
    using Layout = layout::tags::soa_tag;
    if (x.size() != matrix.n_cols()) {
        throw std::invalid_argument(
            "The size of the input vector x does not match the number of columns of the matrix");
    }
    if (y.size() != matrix.n_rows()) {
        throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
    }

    matrix.flush();

    auto SpMV = [&y, &x](const row<Layout, uint32_t, float> &row, uint32_t rowIndex) {
        y[rowIndex] = kernel::sparse_dot_f32(row.data().second.data(), row.data().first.data(), x.data(), row.size());
    };

    matrix.for_each_row(SpMV);
}

template <>
inline void
spmv<layout::tags::soa_tag, uint32_t, double>(const spira::matrix<layout::tags::soa_tag, uint32_t, double> &matrix,
                                             const std::vector<double> &x, std::vector<double> &y) {
    using Layout = layout::tags::soa_tag;
    if (x.size() != matrix.n_cols()) {
        throw std::invalid_argument(
            "The size of the input vector x does not match the number of columns of the matrix");
    }
    if (y.size() != matrix.n_rows()) {
        throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
    }

    matrix.flush();

    auto SpMV = [&y, &x](const row<Layout, uint32_t, double> &row, uint32_t rowIndex) {
        y[rowIndex] = kernel::sparse_dot_f64(row.data().second.data(), row.data().first.data(), x.data(), row.size());
    };

    matrix.for_each_row(SpMV);
}

} // namespace spira::algorithms
