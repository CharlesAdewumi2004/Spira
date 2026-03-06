#pragma once
#include <cassert>
#include <spira/matrix/matrix.hpp>

namespace spira::algorithms {

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void multiplication_scaler(spira::matrix<Layout, I, V> &mat, V scaler) {
    assert(mat.is_open() && "multiplication_scaler: matrix must be open");
    mat.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val *= scaler; }); });
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void multiplication_scaler(const spira::matrix<Layout, I, V> &mat, spira::matrix<Layout, I, V> &out, V scaler) {
    assert(mat.is_locked() && "multiplication_scaler: input matrix must be locked");
    out = mat;
    out.open();
    out.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val *= scaler; }); });
    out.lock();
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void division_scaler(spira::matrix<Layout, I, V> &mat, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler)) {
        throw std::domain_error("Divison by zero");
    }
    assert(mat.is_open() && "division_scaler: matrix must be open");

    mat.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val /= scaler; }); });
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void division_scaler(const spira::matrix<Layout, I, V> &mat, spira::matrix<Layout, I, V> &out, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler)) {
        throw std::domain_error("Divison by zero");
    }
    assert(mat.is_locked() && "division_scaler: input matrix must be locked");

    out = mat;
    out.open();
    out.for_each_row(
        [scaler](auto &row, I /*row_index*/) { row.for_each_element([scaler](I /*col*/, V &val) { val /= scaler; }); });
    out.lock();
}

} // namespace spira::algorithms
