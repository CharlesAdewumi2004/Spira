#pragma once
#include <spira/matrix/matrix.hpp>

namespace spira::algorithms {

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void multiplication_scaler(spira::matrix<Layout, I, V> &mat, V scaler) {
    mat.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val *= scaler; }); });
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void multiplication_scaler(const spira::matrix<Layout, I, V> &mat, spira::matrix<Layout, I, V> &out, V scaler) {
    out = mat;
    out.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val *= scaler; }); });
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void division_scaler(spira::matrix<Layout, I, V> &mat, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler)) {
        throw std::domain_error("Divison by zero");
    }

    mat.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val /= scaler; }); });
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void division_scaler(const spira::matrix<Layout, I, V> &mat, spira::matrix<Layout, I, V> &out, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler)) {
        throw std::domain_error("Divison by zero");
    }

    out = mat;
    out.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val /= scaler; }); });
}

} // namespace spira::algorithms
