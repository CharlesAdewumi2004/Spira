#pragma once
#include <stdexcept>
#include <string>
#include <spira/matrix/matrix.hpp>

namespace spira::serial::algorithms {

namespace detail {

// In place. Pending inserts are committed first so every current value is in
// the CSR, then the matrix is reopened and each scaled value is staged through
// matrix::insert(), which queues its row for the next lock(). Writing to a
// row's buffer while reading its CSR slice is safe: the slice points into the
// matrix's CSR arrays, not the buffer.
template <class M, class Op>
void scale_in_place(M &mat, Op op) {
    using I = typename M::index_type;
    mat.lock();
    mat.open();
    for (std::size_t r = 0; r < mat.n_rows(); ++r) {
        const I row_idx = static_cast<I>(r);
        mat.row_at(row_idx).for_each_element(
            [&mat, &op, row_idx](const I col, const auto val) {
                mat.insert(row_idx, col, op(val));
            });
    }
}

// Copy. out is replaced by a locked matrix of mat's shape holding op(value)
// for every entry of the locked matrix mat.
template <class M, class Op>
void scale_copy(const M &mat, M &out, Op op, const char *name) {
    using I = typename M::index_type;
    if (!mat.is_locked())
        throw std::logic_error(std::string(name) + ": input matrix must be locked");
    out = M(mat.n_rows(), mat.n_cols());
    mat.for_each_row([&out, &op](const auto &in_row, I row_idx) {
        in_row.for_each_element([&out, &op, row_idx](I col, const auto val) {
            out.insert(row_idx, col, op(val));
        });
    });
    out.lock();
}

} // namespace detail

/// In-place scalar multiply. The matrix must be open and stays open; the
/// scaled values are committed by the next lock().
template <class Layout, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, class S>
void multiplication_scaler(spira::matrix<Layout, I, V, BT, BN, S> &mat, V scaler) {
    if (!mat.is_open())
        throw std::logic_error("multiplication_scaler: matrix must be open");
    detail::scale_in_place(mat, [scaler](const V v) { return v * scaler; });
}

/// Copy: out becomes scaler × mat (mat must be locked); out is left locked.
template <class Layout, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, class S>
void multiplication_scaler(const spira::matrix<Layout, I, V, BT, BN, S> &mat, spira::matrix<Layout, I, V, BT, BN, S> &out, V scaler) {
    detail::scale_copy(mat, out, [scaler](const V v) { return v * scaler; }, "multiplication_scaler");
}

template <class Layout, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, class S>
void division_scaler(spira::matrix<Layout, I, V, BT, BN, S> &mat, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler))
        throw std::domain_error("Divison by zero");
    if (!mat.is_open())
        throw std::logic_error("division_scaler: matrix must be open");
    detail::scale_in_place(mat, [scaler](const V v) { return v / scaler; });
}

template <class Layout, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, class S>
void division_scaler(const spira::matrix<Layout, I, V, BT, BN, S> &mat, spira::matrix<Layout, I, V, BT, BN, S> &out, V scaler) {
    if (spira::traits::ValueTraits<V>::is_zero(scaler))
        throw std::domain_error("Divison by zero");
    detail::scale_copy(mat, out, [scaler](const V v) { return v / scaler; }, "division_scaler");
}

} // namespace spira::serial::algorithms
