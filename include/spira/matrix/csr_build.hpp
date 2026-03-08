#pragma once

#include <cstddef>
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

        // ── Pass 1: compute new per-row nnz ──────────────────────────────────

        struct MergedRow
        {
            std::vector<I> cols;
            std::vector<V> vals;
        };
        std::vector<MergedRow> merged(n_rows);

        for (std::size_t i = 0; i < n_rows; ++i)
        {
            const std::size_t old_begin = old_csr.offsets[i];
            const std::size_t old_end = old_csr.offsets[i + 1];
            const std::size_t old_nnz = old_end - old_begin;

            // Buffer iterator (sorted+deduped after row::lock())
            auto bit = rows[i].begin();
            auto bend = rows[i].end();

            auto &mc = merged[i];
            mc.cols.reserve(old_nnz + static_cast<std::size_t>(bend - bit));
            mc.vals.reserve(mc.cols.capacity());

            std::size_t oi = 0;

            auto old_col = [&](std::size_t idx) -> I
            {
                if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                    return old_csr.cols.get()[old_begin + idx];
                else
                    return old_csr.pairs.get()[old_begin + idx].column;
            };
            auto old_val = [&](std::size_t idx) -> V
            {
                if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                    return old_csr.vals.get()[old_begin + idx];
                else
                    return old_csr.pairs.get()[old_begin + idx].value;
            };

            while (oi < old_nnz && bit != bend)
            {
                const I o_col = old_col(oi);
                const I b_col = (*bit).first_ref();

                if (o_col < b_col)
                {
                    if (!traits::ValueTraits<V>::is_zero(old_val(oi)))
                    {
                        mc.cols.push_back(o_col);
                        mc.vals.push_back(old_val(oi));
                    }
                    ++oi;
                }
                else if (b_col < o_col)
                {
                    if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                    {
                        mc.cols.push_back(b_col);
                        mc.vals.push_back((*bit).second_ref());
                    }
                    ++bit;
                }
                else
                {
                    // Same column: buffer wins (more recent write).
                    if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                    {
                        mc.cols.push_back(b_col);
                        mc.vals.push_back((*bit).second_ref());
                    }
                    ++oi;
                    ++bit;
                }
            }
            while (oi < old_nnz)
            {
                if (!traits::ValueTraits<V>::is_zero(old_val(oi)))
                {
                    mc.cols.push_back(old_col(oi));
                    mc.vals.push_back(old_val(oi));
                }
                ++oi;
            }
            while (bit != bend)
            {
                if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                {
                    mc.cols.push_back((*bit).first_ref());
                    mc.vals.push_back((*bit).second_ref());
                }
                ++bit;
            }
        }

        // ── Pass 2: allocate new CSR and copy ────────────────────────────────

        std::size_t total_nnz = 0;
        for (const auto &mc : merged)
            total_nnz += mc.cols.size();

        csr_storage<LayoutTag, I, V> csr(n_rows, total_nnz);
        csr.offsets[0] = 0;
        for (std::size_t i = 0; i < n_rows; ++i)
            csr.offsets[i + 1] = csr.offsets[i] + merged[i].cols.size();

        if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
        {
            I *cols_ptr = csr.cols.get();
            V *vals_ptr = csr.vals.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                const std::size_t k = csr.offsets[i];
                const auto &mc = merged[i];
                for (std::size_t j = 0; j < mc.cols.size(); ++j)
                {
                    cols_ptr[k + j] = mc.cols[j];
                    vals_ptr[k + j] = mc.vals[j];
                }
            }
        }
        else // aos_tag
        {
            auto *pairs_ptr = csr.pairs.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                const std::size_t k = csr.offsets[i];
                const auto &mc = merged[i];
                for (std::size_t j = 0; j < mc.cols.size(); ++j)
                    pairs_ptr[k + j] = {mc.cols[j], mc.vals[j]};
            }
        }

        return csr;
    }

} // namespace spira
