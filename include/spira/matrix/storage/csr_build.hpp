#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include <spira/matrix/layout/element_pair.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/storage/csr_storage.hpp>
#include <spira/traits.hpp>

namespace spira
{

    // ─────────────────────────────────────────────────────────────────────────────
    // CSR construction and incremental re-lock.
    //
    // The locked CSR gives every row a slot with slack (row_slot_capacity), so a
    // re-lock only has to touch the rows that were edited:
    //
    //   build_csr    — first lock: lay every row out from its sorted buffer.
    //   relock_rows  — later locks: merge each dirty row's buffer into its own
    //                  slot. A row that no longer fits moves to the free tail
    //                  and leaves a hole. When the tail is too small or holes
    //                  pass a quarter of the assigned slots, everything is
    //                  repacked into a fresh allocation instead.
    //
    // Zero values in a buffer are deletion signals: they remove the matching
    // committed entry and are never written to the CSR.
    //
    // Row buffers must be sorted and deduplicated (row::lock()) before either
    // function reads them.
    // ─────────────────────────────────────────────────────────────────────────────

    namespace detail
    {
        // Free tail reserved after the assigned slots when a CSR is laid out,
        // so the first rows that outgrow their slot can move without a repack.
        constexpr std::size_t tail_reserve(std::size_t assigned) noexcept
        {
            return assigned / 8;
        }

        // Capacity for a row that has outgrown its slot: normal slack for the
        // new length, and at least double the old slot so a row that keeps
        // growing moves O(log n) times.
        constexpr std::size_t relocated_capacity(std::size_t need, std::size_t old_cap) noexcept
        {
            const std::size_t slack = row_slot_capacity(need);
            return slack > 2 * old_cap ? slack : 2 * old_cap;
        }

        // Allocate a CSR whose rows get row_slot_capacity(len_ub[r]) slots each,
        // in row order, plus a free tail. Row lengths start at zero.
        template <class LayoutTag, class I, class V>
        csr_storage<LayoutTag, I, V> layout_rows(const std::vector<std::size_t> &len_ub)
        {
            const std::size_t n_rows = len_ub.size();
            std::size_t assigned = 0;
            for (std::size_t r = 0; r < n_rows; ++r)
                assigned += row_slot_capacity(len_ub[r]);

            csr_storage<LayoutTag, I, V> csr(n_rows, assigned + tail_reserve(assigned));
            std::size_t pos = 0;
            for (std::size_t r = 0; r < n_rows; ++r)
            {
                csr.row_start[r] = pos;
                csr.row_len[r] = 0;
                csr.row_cap[r] = row_slot_capacity(len_ub[r]);
                pos += csr.row_cap[r];
            }
            csr.end = assigned;
            return csr;
        }

        // Two-pointer merge of a row's committed entries (old_count of them,
        // read through old_col/old_val) with its sorted buffer, written to
        // consecutive slots from dst. The buffer wins on a shared column; zeros
        // are dropped. Returns the number of entries written.
        template <class LayoutTag, class I, class V, class RowType, class OldCol, class OldVal>
        std::size_t merge_row(csr_storage<LayoutTag, I, V> &out, std::size_t dst,
                              const RowType &row, std::size_t old_count,
                              OldCol &&old_col, OldVal &&old_val)
        {
            std::size_t wp = dst;
            auto emit = [&](I c, const V &v)
            {
                if (!traits::ValueTraits<V>::is_zero(v))
                    out.set(wp++, c, v);
            };

            auto bit = row.begin();
            const auto bend = row.end();
            std::size_t oi = 0;
            while (oi < old_count && bit != bend)
            {
                const I oc = old_col(oi);
                const I bc = (*bit).first_ref();
                if (oc < bc)
                {
                    emit(oc, old_val(oi));
                    ++oi;
                }
                else
                {
                    emit(bc, (*bit).second_ref());
                    if (oc == bc)
                        ++oi;
                    ++bit;
                }
            }
            for (; oi < old_count; ++oi)
                emit(old_col(oi), old_val(oi));
            for (; bit != bend; ++bit)
                emit((*bit).first_ref(), (*bit).second_ref());
            return wp - dst;
        }

        template <class RowType>
        std::size_t buffer_size(const RowType &row) noexcept
        {
            return static_cast<std::size_t>(row.end() - row.begin());
        }

        // Rebuild every row into a fresh allocation: dirty rows are merged
        // with their buffers, clean rows are copied. Used when a re-lock would
        // overflow the free tail or leave too many holes.
        template <class LayoutTag, class RowType>
        auto repack(const csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type> &old,
                    const std::vector<RowType> &rows, const std::vector<bool> &is_dirty)
            -> csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type>
        {
            using I = typename RowType::index_type;
            using V = typename RowType::value_type;

            const std::size_t n_rows = rows.size();
            std::vector<std::size_t> len_ub(n_rows);
            for (std::size_t r = 0; r < n_rows; ++r)
                len_ub[r] = old.row_len[r] + (is_dirty[r] ? buffer_size(rows[r]) : 0);

            auto out = layout_rows<LayoutTag, I, V>(len_ub);
            for (std::size_t r = 0; r < n_rows; ++r)
            {
                const std::size_t src = old.row_start[r];
                const std::size_t len = old.row_len[r];
                if (!is_dirty[r])
                {
                    out.copy_from(out.row_start[r], old, src, len);
                    out.row_len[r] = len;
                }
                else
                {
                    out.row_len[r] = merge_row(
                        out, out.row_start[r], rows[r], len,
                        [&](std::size_t k) { return old.col(src + k); },
                        [&](std::size_t k) { return old.val(src + k); });
                }
                out.nnz += out.row_len[r];
            }
            return out;
        }
    } // namespace detail

    // ─────────────────────────────────────────────────────────────────────────────
    // build_csr<LayoutTag>
    //
    // First-lock construction from rows whose buffers are sorted and
    // deduplicated. Zero values are skipped (there is nothing to delete yet).
    // Precondition: every row in `rows` is locked.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class RowType>
    auto build_csr(const std::vector<RowType> &rows)
        -> csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type>
    {
        using I = typename RowType::index_type;
        using V = typename RowType::value_type;

        const std::size_t n_rows = rows.size();
        std::vector<std::size_t> len(n_rows);
        for (std::size_t r = 0; r < n_rows; ++r)
        {
            std::size_t cnt = 0;
            for (const auto &entry : rows[r])
                if (!traits::ValueTraits<V>::is_zero(entry.second_ref()))
                    ++cnt;
            len[r] = cnt;
        }

        auto csr = detail::layout_rows<LayoutTag, I, V>(len);
        for (std::size_t r = 0; r < n_rows; ++r)
        {
            csr.row_len[r] = detail::merge_row(
                csr, csr.row_start[r], rows[r], 0,
                [](std::size_t) { return I{}; }, [](std::size_t) { return V{}; });
            csr.nnz += csr.row_len[r];
        }
        return csr;
    }

    /// Point every row at its slot in csr.
    template <class LayoutTag, class RowType>
    void install_slices(const csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type> &csr,
                        std::vector<RowType> &rows)
    {
        for (std::size_t r = 0; r < rows.size(); ++r)
            rows[r].set_csr_slice(csr.slice(r));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // relock_rows<LayoutTag>
    //
    // Commit the buffers of the rows listed in `dirty` (each listed once) into
    // an already-built CSR. Rows not listed are not read or written unless a
    // repack is needed. On return every row's CSR slice is valid and the
    // listed rows' buffers are cleared.
    //
    // Returns true if the CSR was repacked (every row moved).
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class RowType>
    bool relock_rows(csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type> &csr,
                     std::vector<RowType> &rows, const std::vector<std::size_t> &dirty)
    {
        using I = typename RowType::index_type;
        using V = typename RowType::value_type;

        // Sort each dirty buffer, and total the tail space and holes that
        // moving the overflowing rows would need.
        std::size_t tail_needed = 0;
        std::size_t freed = 0;
        for (const std::size_t r : dirty)
        {
            rows[r].lock();
            const std::size_t need = csr.row_len[r] + detail::buffer_size(rows[r]);
            if (need > csr.row_cap[r])
            {
                tail_needed += detail::relocated_capacity(need, csr.row_cap[r]);
                freed += csr.row_cap[r];
            }
        }

        const std::size_t end_after = csr.end + tail_needed;
        const bool out_of_tail = end_after > csr.capacity;
        const bool too_many_holes = 4 * (csr.holes + freed) > end_after;

        if (out_of_tail || too_many_holes)
        {
            std::vector<bool> is_dirty(rows.size(), false);
            for (const std::size_t r : dirty)
                is_dirty[r] = true;
            csr = detail::repack<LayoutTag>(csr, rows, is_dirty);
            install_slices<LayoutTag>(csr, rows);
            for (const std::size_t r : dirty)
                rows[r].clear_buffer_content();
            return true;
        }

        // Every row fits in its slot or in the tail: merge one row at a time.
        // Merging in place would overwrite committed entries before they are
        // read, so each row's old entries are copied to scratch first.
        thread_local std::vector<I> old_cols;
        thread_local std::vector<V> old_vals;
        for (const std::size_t r : dirty)
        {
            const std::size_t len = csr.row_len[r];
            const std::size_t start = csr.row_start[r];
            old_cols.resize(len);
            old_vals.resize(len);
            for (std::size_t k = 0; k < len; ++k)
            {
                old_cols[k] = csr.col(start + k);
                old_vals[k] = csr.val(start + k);
            }

            const std::size_t need = len + detail::buffer_size(rows[r]);
            if (need > csr.row_cap[r])
            {
                const std::size_t cap = detail::relocated_capacity(need, csr.row_cap[r]);
                csr.holes += csr.row_cap[r];
                csr.row_start[r] = csr.end;
                csr.row_cap[r] = cap;
                csr.end += cap;
            }

            const std::size_t new_len = detail::merge_row(
                csr, csr.row_start[r], rows[r], len,
                [](std::size_t k) { return old_cols[k]; },
                [](std::size_t k) { return old_vals[k]; });
            csr.nnz = csr.nnz - len + new_len;
            csr.row_len[r] = new_len;

            rows[r].set_csr_slice(csr.slice(r));
            rows[r].clear_buffer_content();
        }
        return false;
    }

} // namespace spira
