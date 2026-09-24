#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/matrix/storage/csr_build.hpp>
#include <spira/matrix/storage/csr_storage.hpp>
#include <spira/matrix/layout/layout_tags.hpp>

namespace spira::serial::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────
    // transpose — two-pass histogram + direct fill, O(nnz), zero sorting.
    //
    // The input is locked, so every row is a contiguous run in the CSR arrays.
    // Both passes walk rows through row_start/row_len (the slack between rows
    // holds stale data and must not be read). This matches Eigen's two-pass
    // implementation.
    //
    // Since input rows are processed in ascending order i = 0..r-1, output
    // rows receive entries with column = i in monotonically increasing order
    // → output CSR is already sorted. The output is laid out with the usual
    // per-row slack, and load_csr() installs it with no sort.
    //
    // mat must be locked.
    // ─────────────────────────────────────────────────────────────────────────

    template <class Layout, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN>
    spira::matrix<Layout, I, V, BT, BN> transpose(const spira::matrix<Layout, I, V, BT, BN> &mat)
    {
        if (!mat.is_locked())
            throw std::logic_error("transpose: matrix must be locked");

        const auto [r, c] = mat.shape();
        const auto *in_csr = mat.csr(); // always built once locked
        const std::size_t *in_start = in_csr->row_start.get();
        const std::size_t *in_len = in_csr->row_len.get();

        // Pass 1: histogram of column indices.
        std::vector<std::size_t> counts(c, 0);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = in_start[i]; k < in_start[i] + in_len[i]; ++k)
                counts[static_cast<std::size_t>(in_csr->col(k))]++;

        auto out_csr = spira::detail::layout_rows<Layout, I, V>(counts);
        std::vector<std::size_t> cursor(c);
        for (std::size_t j = 0; j < c; ++j)
        {
            cursor[j] = out_csr.row_start[j];
            out_csr.row_len[j] = counts[j];
            out_csr.nnz += counts[j];
        }

        // Pass 2: fill, reading input rows in order.
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t k = in_start[i]; k < in_start[i] + in_len[i]; ++k)
                out_csr.set(cursor[static_cast<std::size_t>(in_csr->col(k))]++,
                            static_cast<I>(i), in_csr->val(k));

        spira::matrix<Layout, I, V, BT, BN> result(c, r);
        result.load_csr(std::move(out_csr));
        return result;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN>
    void transpose_itself(spira::matrix<Layout, I, V, BT, BN> &mat)
    {
        if (!mat.is_open())
            throw std::logic_error("transpose_itself: matrix must be open");

        auto [r, c] = mat.shape();
        if (r != c)
            throw std::logic_error("in-place transpose requires square matrix");

        mat.lock();
        auto out = transpose(mat);
        mat.swap(out);
        mat.open();
    }

} // namespace spira::serial::algorithms
