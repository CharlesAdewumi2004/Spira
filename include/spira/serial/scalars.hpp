#pragma once
#include <stdexcept>
#include <spira/matrix/matrix.hpp>

namespace spira::serial::algorithms {

// In-place scalar multiply: reads committed CSR entries, inserts scaled values
// into the buffer. Subsequent lock() will merge-overwrite with the new values.
template <class Layout, concepts::Indexable I, concepts::Valueable V>
void multiplication_scaler(spira::matrix<Layout, I, V> &mat, V scaler) {
    if (!mat.is_open())
        throw std::logic_error("multiplication_scaler: matrix must be open");
    mat.for_each_row([scaler](auto &row, I /*row_index*/) {
        row.for_each_committed_element([&row, scaler](const I col, const V val) {
            row.insert(col, val * scaler);
        });
    });
}

// Copy path: build result from scratch into out using matrix::insert() so
// dirty_ flags are set correctly for the subsequent lock()/merge_csr() call.
template <class Layout, concepts::Indexable I, concepts::Valueable V>
void multiplication_scaler(const spira::matrix<Layout, I, V> &mat,
                            spira::matrix<Layout, I, V> &out, V scaler) {
    if (!mat.is_locked())
        throw std::logic_error("multiplication_scaler: input matrix must be locked");
    const auto [rows, cols] = mat.shape();
    out = spira::matrix<Layout, I, V>(rows, cols);
    mat.for_each_row([&out, scaler](const auto &in_row, I row_idx) {
        in_row.for_each_element([&out, scaler, row_idx](I col, V val) {
            out.insert(row_idx, col, val * scaler);
        });
    });
    out.lock();
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void division_scaler(spira::matrix<Layout, I, V> &mat, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler)) {
        throw std::domain_error("Divison by zero");
    }
    if (!mat.is_open())
        throw std::logic_error("division_scaler: matrix must be open");
    mat.for_each_row([scaler](auto &row, I /*row_index*/) {
        row.for_each_committed_element([&row, scaler](const I col, const V val) {
            row.insert(col, val / scaler);
        });
    });
}

template <class Layout, concepts::Indexable I, concepts::Valueable V>
void division_scaler(const spira::matrix<Layout, I, V> &mat,
                      spira::matrix<Layout, I, V> &out, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler)) {
        throw std::domain_error("Divison by zero");
    }
    if (!mat.is_locked())
        throw std::logic_error("division_scaler: input matrix must be locked");
    const auto [rows, cols] = mat.shape();
    out = spira::matrix<Layout, I, V>(rows, cols);
    mat.for_each_row([&out, scaler](const auto &in_row, I row_idx) {
        in_row.for_each_element([&out, scaler, row_idx](I col, V val) {
            out.insert(row_idx, col, val / scaler);
        });
    });
    out.lock();
}

} // namespace spira::algorithms
