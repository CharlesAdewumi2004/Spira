#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

#include <spira/matrix/csr_storage.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/traits.hpp>

namespace spira
{

    // ─────────────────────────────────────────────────────────────────────────────
    // build_csr<LayoutTag>
    //
    // First-lock construction: two-pass build from a vector of locked rows whose
    // buffers have been sorted+deduped+filtered by row::lock().
    //
    //   Pass 1 — walk rows[i].size() to fill offsets[] and sum total nnz.
    //   Pass 2 — iterate each row's sorted buffer and copy to the CSR arrays.
    //            For soa_tag: fills cols[] and vals[] separately.
    //            For aos_tag: fills pairs[] with interleaved {col, val} entries.
    //
    // Zero values are already filtered by row::lock(); no extra filtering needed.
    // Precondition: every row in `rows` must be in locked mode.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class RowType>
    auto build_csr(const std::vector<RowType> &rows)
        -> csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type>
    {
        using I = typename RowType::index_type;
        using V = typename RowType::value_type;

        const std::size_t n_rows = rows.size();

        std::size_t total_nnz = 0;
        for (std::size_t i = 0; i < n_rows; ++i)
            total_nnz += rows[i].size();

        csr_storage<LayoutTag, I, V> csr(n_rows, total_nnz);

        csr.offsets[0] = 0;
        for (std::size_t i = 0; i < n_rows; ++i)
            csr.offsets[i + 1] = csr.offsets[i] + rows[i].size();

        if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
        {
            I *cols_ptr = csr.cols.get();
            V *vals_ptr = csr.vals.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                std::size_t k = csr.offsets[i];
                for (const auto &entry : rows[i])
                {
                    cols_ptr[k] = entry.first_ref();
                    vals_ptr[k] = entry.second_ref();
                    ++k;
                }
            }
        }
        else // aos_tag
        {
            auto *pairs_ptr = csr.pairs.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                std::size_t k = csr.offsets[i];
                for (const auto &entry : rows[i])
                {
                    pairs_ptr[k] = {entry.first_ref(), entry.second_ref()};
                    ++k;
                }
            }
        }

        return csr;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // merge_csr<LayoutTag>
    //
    // Subsequent-lock merge: for each row, performs a two-pointer merge of the
    // existing CSR slice (committed history) with the row's sorted buffer delta
    // (new inserts since last open()). Buffer wins on column collision (newer
    // write replaces older committed value). Zero values are filtered.
    //
    // Allocation strategy: pre-compute an upper-bound total nnz (old nnz + all
    // buffer entries), allocate one output CSR at that size, then merge each row
    // directly into it at its upper-bound write position. A single front-to-back
    // memmove pass compacts any gaps left by zero-filtered entries. This reduces
    // heap allocations from O(n_rows) to O(1), regardless of matrix size.
    //
    // Precondition: every row in `rows` must be in locked mode.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class RowType>
    auto merge_csr(const std::vector<RowType> &rows,
                   const csr_storage<LayoutTag,
                                     typename RowType::index_type,
                                     typename RowType::value_type> &old_csr)
        -> csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type>
    {
        using I = typename RowType::index_type;
        using V = typename RowType::value_type;

        const std::size_t n_rows = rows.size();

        // Upper-bound nnz: every old entry + every new buffer entry.
        // The actual count can only be equal or smaller (zero-filtered writes reduce it).
        std::size_t total_ub = old_csr.nnz;
        for (std::size_t i = 0; i < n_rows; ++i)
            total_ub += static_cast<std::size_t>(rows[i].end() - rows[i].begin());

        // Single allocation for the output CSR at upper-bound size.
        csr_storage<LayoutTag, I, V> out(n_rows, total_ub);

        // Upper-bound write start per row (one alloc, stays on the stack for small n_rows
        // but heap for large matrices — still O(1) allocations total).
        auto ub_starts = std::make_unique<std::size_t[]>(n_rows);
        {
            std::size_t pos = 0;
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                ub_starts[i] = pos;
                pos += (old_csr.offsets[i + 1] - old_csr.offsets[i])
                     + static_cast<std::size_t>(rows[i].end() - rows[i].begin());
            }
        }

        // ── Merge pass: write each row directly into out at ub_starts[i] ──────

        out.offsets[0] = 0;

        for (std::size_t i = 0; i < n_rows; ++i)
        {
            const std::size_t old_begin = old_csr.offsets[i];
            const std::size_t old_nnz   = old_csr.offsets[i + 1] - old_begin;

            auto bit  = rows[i].begin();
            auto bend = rows[i].end();

            auto old_col = [&](std::size_t k) -> I
            {
                if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                    return old_csr.cols.get()[old_begin + k];
                else
                    return old_csr.pairs.get()[old_begin + k].column;
            };
            auto old_val = [&](std::size_t k) -> V
            {
                if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                    return old_csr.vals.get()[old_begin + k];
                else
                    return old_csr.pairs.get()[old_begin + k].value;
            };

            std::size_t wp = ub_starts[i]; // write position in out

            auto emit = [&](I col, V val)
            {
                if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                {
                    out.cols.get()[wp] = col;
                    out.vals.get()[wp] = val;
                }
                else
                {
                    out.pairs.get()[wp] = {col, val};
                }
                ++wp;
            };

            std::size_t oi = 0;
            while (oi < old_nnz && bit != bend)
            {
                const I oc = old_col(oi);
                const I bc = (*bit).first_ref();
                if (oc < bc)
                {
                    if (!traits::ValueTraits<V>::is_zero(old_val(oi))) emit(oc, old_val(oi));
                    ++oi;
                }
                else if (bc < oc)
                {
                    if (!traits::ValueTraits<V>::is_zero((*bit).second_ref())) emit(bc, (*bit).second_ref());
                    ++bit;
                }
                else
                {
                    // Same column: buffer wins (more recent write).
                    if (!traits::ValueTraits<V>::is_zero((*bit).second_ref())) emit(bc, (*bit).second_ref());
                    ++oi;
                    ++bit;
                }
            }
            while (oi < old_nnz)
            {
                if (!traits::ValueTraits<V>::is_zero(old_val(oi))) emit(old_col(oi), old_val(oi));
                ++oi;
            }
            while (bit != bend)
            {
                if (!traits::ValueTraits<V>::is_zero((*bit).second_ref())) emit((*bit).first_ref(), (*bit).second_ref());
                ++bit;
            }

            out.offsets[i + 1] = out.offsets[i] + (wp - ub_starts[i]);
        }

        out.nnz = out.offsets[n_rows];

        // ── Compact pass: close gaps left by zero-filtered entries ─────────────
        //
        // actual_start[i] = out.offsets[i]  ≤  ub_starts[i] = upper-bound start.
        // Processing front-to-back is safe: each row's destination always ends
        // before the next row's upper-bound source starts.

        if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
        {
            I *cols_ptr = out.cols.get();
            V *vals_ptr = out.vals.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                const std::size_t dst = out.offsets[i];
                const std::size_t src = ub_starts[i];
                const std::size_t cnt = out.offsets[i + 1] - dst;
                if (src != dst && cnt > 0)
                {
                    std::memmove(cols_ptr + dst, cols_ptr + src, cnt * sizeof(I));
                    std::memmove(vals_ptr + dst, vals_ptr + src, cnt * sizeof(V));
                }
            }
        }
        else // aos_tag
        {
            using P = layout::elementPair<I, V>;
            auto *pairs_ptr = out.pairs.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                const std::size_t dst = out.offsets[i];
                const std::size_t src = ub_starts[i];
                const std::size_t cnt = out.offsets[i + 1] - dst;
                if (src != dst && cnt > 0)
                    std::memmove(pairs_ptr + dst, pairs_ptr + src, cnt * sizeof(P));
            }
        }

        return out;
    }

} // namespace spira
