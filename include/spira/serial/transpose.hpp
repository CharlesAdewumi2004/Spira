#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/matrix/storage/csr_storage.hpp>
#include <spira/matrix/layout/layout_tags.hpp>

namespace spira::serial::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────
    // transpose — two-pass histogram + direct fill, O(nnz), zero sorting.
    //
    // The input is locked, so it has a flat CSR slab. Both passes scan the raw
    // cols[]/vals[] (SoA) or pairs[] (AoS) arrays directly — no per-row function calls, no virtual dispatch,
    // fully sequential reads. The histogram pass is a single linear scan of
    // the column-index array; the fill pass is row-by-row but still reads
    // flat pointers. This matches Eigen's two-pass implementation.
    //
    // Since input rows are processed in ascending order i = 0..r-1, output
    // rows receive entries with column = i in monotonically increasing order
    // → output CSR is already sorted. load_csr() installs it with no sort.
    //
    // mat must be locked.
    // ─────────────────────────────────────────────────────────────────────────

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    spira::matrix<Layout, I, V> transpose(const spira::matrix<Layout, I, V> &mat)
    {
        if (!mat.is_locked())
            throw std::logic_error("transpose: matrix must be locked");

        const auto [r, c] = mat.shape();
        const std::size_t total_nnz = mat.nnz();

        std::vector<std::size_t> counts(c, 0);
        csr_storage<Layout, I, V> out_csr(c, total_nnz);

        const auto *in_csr = mat.csr(); // always built once locked

        // Pass 1: histogram — single sequential scan of column indices.
        if constexpr (std::is_same_v<Layout, layout::tags::soa_tag>)
        {
            const I *cols_flat = in_csr->cols.get();
            for (std::size_t k = 0; k < total_nnz; ++k)
                counts[static_cast<std::size_t>(cols_flat[k])]++;
        }
        else
        {
            const auto *pairs_flat = in_csr->pairs.get();
            for (std::size_t k = 0; k < total_nnz; ++k)
                counts[static_cast<std::size_t>(pairs_flat[k].column)]++;
        }

        // Build offsets + cursor.
        out_csr.offsets[0] = 0;
        for (std::size_t j = 0; j < c; ++j)
            out_csr.offsets[j + 1] = out_csr.offsets[j] + counts[j];

        std::vector<std::size_t> cursor(c);
        for (std::size_t j = 0; j < c; ++j)
            cursor[j] = out_csr.offsets[j];

        // Pass 2: fill — read flat input arrays row-by-row.
        const std::size_t *offsets_in = in_csr->offsets.get();
        if constexpr (std::is_same_v<Layout, layout::tags::soa_tag>)
        {
            const I *cols_in = in_csr->cols.get();
            const V *vals_in = in_csr->vals.get();
            I       *cols_out = out_csr.cols.get();
            V       *vals_out = out_csr.vals.get();

            for (std::size_t i = 0; i < r; ++i)
            {
                const I        ri    = static_cast<I>(i);
                const std::size_t beg = offsets_in[i];
                const std::size_t end = offsets_in[i + 1];
                for (std::size_t k = beg; k < end; ++k)
                {
                    const std::size_t pos = cursor[static_cast<std::size_t>(cols_in[k])]++;
                    cols_out[pos] = ri;
                    vals_out[pos] = vals_in[k];
                }
            }
        }
        else
        {
            const auto *pairs_in  = in_csr->pairs.get();
            auto       *pairs_out = out_csr.pairs.get();

            for (std::size_t i = 0; i < r; ++i)
            {
                const I        ri    = static_cast<I>(i);
                const std::size_t beg = offsets_in[i];
                const std::size_t end = offsets_in[i + 1];
                for (std::size_t k = beg; k < end; ++k)
                {
                    const std::size_t pos = cursor[static_cast<std::size_t>(pairs_in[k].column)]++;
                    pairs_out[pos] = {ri, pairs_in[k].value};
                }
            }
        }
    


        spira::matrix<Layout, I, V> result(c, r);
        result.load_csr(std::move(out_csr));
        return result;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void transpose_itself(spira::matrix<Layout, I, V> &mat)
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
