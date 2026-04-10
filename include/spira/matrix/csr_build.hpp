#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
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
    // existing CSR slice (committed history) with the row's sorted buffer delta.
    // Buffer wins on column collision. Zero values are filtered.
    //
    // Allocation strategy:
    //
    //   total_ub = old_csr.nnz + Σ buf_nnz[i]  (upper bound; zeros may reduce it)
    //
    //   Reuse  (total_ub <= old.capacity):
    //     Merge rows in reverse order directly into the existing allocation so
    //     that writes never reach unread old data. One extra alloc for ub_starts
    //     and one for actual_nnz. CSR arrays are reused (zero new array allocs).
    //
    //   Grow   (total_ub > old.capacity):
    //     Allocate a new CSR at capacity = total_ub × 1.5 to amortise future
    //     growth. Forward merge into the new allocation. Compact with memmove.
    //
    // Takes old_csr by move: the caller gives up ownership. In the reuse path
    // the same storage is returned. In the grow path the old storage is freed.
    //
    // Precondition: every row in `rows` must be in locked mode.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class RowType>
    auto merge_csr(
        const std::vector<RowType> &rows,
        csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type> &&old_csr,
        const std::vector<bool> &dirty)
        -> csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type>
    {
        using I = typename RowType::index_type;
        using V = typename RowType::value_type;
        using Csr = csr_storage<LayoutTag, I, V>;

        const std::size_t n_rows = rows.size();

        // Upper-bound nnz: every old entry + every new buffer entry.
        std::size_t total_ub = old_csr.nnz;
        for (std::size_t i = 0; i < n_rows; ++i)
            total_ub += static_cast<std::size_t>(rows[i].end() - rows[i].begin());

        const bool reuse = (total_ub <= old_csr.capacity);
        const std::size_t new_cap = reuse ? old_csr.capacity : total_ub + total_ub / 2; // 1.5× growth

        // ub_starts[i] = upper-bound write offset for row i in the output arrays.
        auto ub_starts = std::make_unique<std::size_t[]>(n_rows);
        {
            std::size_t pos = 0;
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                ub_starts[i] = pos;
                pos += (old_csr.offsets[i + 1] - old_csr.offsets[i]) + static_cast<std::size_t>(rows[i].end() - rows[i].begin());
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Reuse path — in-place reverse merge
        //
        // Moving old_csr into `out` makes out.cols/vals == old data. We read
        // and write into the same buffer, which is safe with reverse row order:
        // row i writes to [ub_starts[i], ub_starts[i+1]) and reads from
        // [old.offsets[i], old.offsets[i+1]).  Since ub_starts[j] >= old.offsets[j]
        // for all j, processing i = n-1 first ensures writes for row i never
        // reach old data for rows 0..i-1, which are processed later.
        // ─────────────────────────────────────────────────────────────────────
        if (reuse)
        {
            Csr out = std::move(old_csr); // buffers transferred; out.offsets = old boundaries

            // Track actual nnz per row (computed during the reverse merge pass).
            auto actual_nnz = std::make_unique<std::size_t[]>(n_rows);

            // Scratch buffers for the aliasing case (see below).
            thread_local std::vector<I> tl_old_c;
            thread_local std::vector<V> tl_old_v;
            thread_local std::vector<layout::elementPair<I, V>> tl_old_p;

            for (std::size_t ri = 0; ri < n_rows; ++ri)
            {
                const std::size_t i = n_rows - 1 - ri; // reverse order

                const std::size_t old_begin = out.offsets[i];
                const std::size_t old_nnz = out.offsets[i + 1] - old_begin;

                // Clean row: buffer is empty; bulk-move old data to its ub position.
                // The compaction pass will then shift it to the final dense offset.
                if (!dirty[i])
                {
                    if (old_nnz > 0 && ub_starts[i] != old_begin)
                    {
                        if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                        {
                            std::memmove(out.cols.get() + ub_starts[i], out.cols.get() + old_begin, old_nnz * sizeof(I));
                            std::memmove(out.vals.get() + ub_starts[i], out.vals.get() + old_begin, old_nnz * sizeof(V));
                        }
                        else
                        {
                            using P = layout::elementPair<I, V>;
                            std::memmove(out.pairs.get() + ub_starts[i], out.pairs.get() + old_begin, old_nnz * sizeof(P));
                        }
                    }
                    actual_nnz[i] = old_nnz;
                    continue;
                }

                auto bit = rows[i].begin();
                auto bend = rows[i].end();

                // Aliasing guard: when ub_starts[i] == old_begin (no prior row has
                // buffer entries, so write cursor starts at the same position as the
                // old-data read cursor), emitting buffer entries before old entries
                // will overwrite unread old data in-place.  Copy old row data to a
                // thread-local scratch buffer so reads are safe.
                const bool aliased = (ub_starts[i] == old_begin) && (bit != bend) && (old_nnz > 0);
                if (aliased)
                {
                    if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                    {
                        tl_old_c.assign(out.cols.get() + old_begin, out.cols.get() + old_begin + old_nnz);
                        tl_old_v.assign(out.vals.get() + old_begin, out.vals.get() + old_begin + old_nnz);
                    }
                    else
                    {
                        tl_old_p.assign(out.pairs.get() + old_begin, out.pairs.get() + old_begin + old_nnz);
                    }
                }

                auto old_col = [&](std::size_t k) -> I
                {
                    if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                        return aliased ? tl_old_c[k] : out.cols.get()[old_begin + k];
                    else
                        return aliased ? tl_old_p[k].column : out.pairs.get()[old_begin + k].column;
                };
                auto old_val = [&](std::size_t k) -> V
                {
                    if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                        return aliased ? tl_old_v[k] : out.vals.get()[old_begin + k];
                    else
                        return aliased ? tl_old_p[k].value : out.pairs.get()[old_begin + k].value;
                };

                std::size_t wp = ub_starts[i];
                std::size_t oi = 0;

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

                while (oi < old_nnz && bit != bend)
                {
                    const I oc = old_col(oi);
                    const I bc = (*bit).first_ref();
                    if (oc < bc)
                    {
                        if (!traits::ValueTraits<V>::is_zero(old_val(oi)))
                            emit(oc, old_val(oi));
                        ++oi;
                    }
                    else if (bc < oc)
                    {
                        if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                            emit(bc, (*bit).second_ref());
                        ++bit;
                    }
                    else
                    {
                        if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                            emit(bc, (*bit).second_ref());
                        ++oi;
                        ++bit;
                    }
                }
                while (oi < old_nnz)
                {
                    if (!traits::ValueTraits<V>::is_zero(old_val(oi)))
                        emit(old_col(oi), old_val(oi));
                    ++oi;
                }
                while (bit != bend)
                {
                    if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                        emit((*bit).first_ref(), (*bit).second_ref());
                    ++bit;
                }

                actual_nnz[i] = wp - ub_starts[i];
            }

            // Build actual offsets from per-row counts.
            out.offsets[0] = 0;
            for (std::size_t i = 0; i < n_rows; ++i)
                out.offsets[i + 1] = out.offsets[i] + actual_nnz[i];
            out.nnz = out.offsets[n_rows];

            // Compact: shift each row's data from its ub_start to its actual offset.
            // Front-to-back is safe: actual_offset[i] <= ub_starts[i] always.
            if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
            {
                I *cp = out.cols.get();
                V *vp = out.vals.get();
                for (std::size_t i = 0; i < n_rows; ++i)
                {
                    const std::size_t dst = out.offsets[i];
                    const std::size_t src = ub_starts[i];
                    const std::size_t cnt = actual_nnz[i];
                    if (src != dst && cnt > 0)
                    {
                        std::memmove(cp + dst, cp + src, cnt * sizeof(I));
                        std::memmove(vp + dst, vp + src, cnt * sizeof(V));
                    }
                }
            }
            else
            {
                using P = layout::elementPair<I, V>;
                auto *pp = out.pairs.get();
                for (std::size_t i = 0; i < n_rows; ++i)
                {
                    const std::size_t dst = out.offsets[i];
                    const std::size_t src = ub_starts[i];
                    const std::size_t cnt = actual_nnz[i];
                    if (src != dst && cnt > 0)
                        std::memmove(pp + dst, pp + src, cnt * sizeof(P));
                }
            }

            return out;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Grow path — allocate new CSR at 1.5× capacity, forward merge
        // ─────────────────────────────────────────────────────────────────────

        Csr out(n_rows, 0, new_cap);
        out.offsets[0] = 0;

        for (std::size_t i = 0; i < n_rows; ++i)
        {
            const std::size_t old_begin = old_csr.offsets[i];
            const std::size_t old_nnz = old_csr.offsets[i + 1] - old_begin;

            // Clean row: buffer is empty; bulk-copy old data to new allocation.
            if (!dirty[i])
            {
                const std::size_t wp = ub_starts[i];
                if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
                {
                    std::memcpy(out.cols.get() + wp, old_csr.cols.get() + old_begin, old_nnz * sizeof(I));
                    std::memcpy(out.vals.get() + wp, old_csr.vals.get() + old_begin, old_nnz * sizeof(V));
                }
                else
                {
                    using P = layout::elementPair<I, V>;
                    std::memcpy(out.pairs.get() + wp, old_csr.pairs.get() + old_begin, old_nnz * sizeof(P));
                }
                out.offsets[i + 1] = out.offsets[i] + old_nnz;
                continue;
            }

            auto bit = rows[i].begin();
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

            std::size_t wp = ub_starts[i];
            std::size_t oi = 0;

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

            while (oi < old_nnz && bit != bend)
            {
                const I oc = old_col(oi);
                const I bc = (*bit).first_ref();
                if (oc < bc)
                {
                    if (!traits::ValueTraits<V>::is_zero(old_val(oi)))
                        emit(oc, old_val(oi));
                    ++oi;
                }
                else if (bc < oc)
                {
                    if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                        emit(bc, (*bit).second_ref());
                    ++bit;
                }
                else
                {
                    if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                        emit(bc, (*bit).second_ref());
                    ++oi;
                    ++bit;
                }
            }
            while (oi < old_nnz)
            {
                if (!traits::ValueTraits<V>::is_zero(old_val(oi)))
                    emit(old_col(oi), old_val(oi));
                ++oi;
            }
            while (bit != bend)
            {
                if (!traits::ValueTraits<V>::is_zero((*bit).second_ref()))
                    emit((*bit).first_ref(), (*bit).second_ref());
                ++bit;
            }

            out.offsets[i + 1] = out.offsets[i] + (wp - ub_starts[i]);
        }

        out.nnz = out.offsets[n_rows];

        // Compact: close gaps from zero-filtered entries.
        if constexpr (std::is_same_v<LayoutTag, layout::tags::soa_tag>)
        {
            I *cp = out.cols.get();
            V *vp = out.vals.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                const std::size_t dst = out.offsets[i];
                const std::size_t src = ub_starts[i];
                const std::size_t cnt = out.offsets[i + 1] - dst;
                if (src != dst && cnt > 0)
                {
                    std::memmove(cp + dst, cp + src, cnt * sizeof(I));
                    std::memmove(vp + dst, vp + src, cnt * sizeof(V));
                }
            }
        }
        else
        {
            using P = layout::elementPair<I, V>;
            auto *pp = out.pairs.get();
            for (std::size_t i = 0; i < n_rows; ++i)
            {
                const std::size_t dst = out.offsets[i];
                const std::size_t src = ub_starts[i];
                const std::size_t cnt = out.offsets[i + 1] - dst;
                if (src != dst && cnt > 0)
                    std::memmove(pp + dst, pp + src, cnt * sizeof(P));
            }
        }

        return out;
    }

} // namespace spira
