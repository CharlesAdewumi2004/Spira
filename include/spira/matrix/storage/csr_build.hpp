#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include <spira/config.hpp>
#include <spira/matrix/layout/element_pair.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/storage/csr_storage.hpp>
#include <spira/traits.hpp>

namespace spira
{

    // ─────────────────────────────────────────────────────────────────────────────
    // CSR construction and incremental re-lock.
    //
    // The locked CSR gives every row a slot with slack (a config::row_slack
    // policy, the Slack parameter below), so a re-lock only has to touch the
    // rows that were edited:
    //
    //   build_csr    — first lock: lay every row out from its sorted buffer.
    //   relock_rows  — later locks: merge each dirty row's buffer into its own
    //                  slot. A row that no longer fits moves to the free tail
    //                  and leaves a hole. When the tail is too small or holes
    //                  pass Slack's repack threshold, everything is repacked
    //                  into a fresh allocation instead.
    //
    // Zero values in a buffer are deletion signals: they remove the matching
    // committed entry and are never written to the CSR.
    //
    // Row buffers must be sorted and deduplicated (row::lock()) before either
    // function reads them.
    // ─────────────────────────────────────────────────────────────────────────────

    namespace detail
    {
        // Capacity for a row that has outgrown its slot. The first time, it
        // gets the policy's normal slack for its new length; a row that grows
        // again gets at least double its old slot, so a row that keeps
        // growing moves O(log n) times.
        template <class Slack>
        constexpr std::size_t relocated_capacity(std::size_t need, std::size_t old_cap, bool grown_before) noexcept
        {
            const std::size_t slack = Slack::capacity(need);
            if (!grown_before)
                return slack;
            return slack > 2 * old_cap ? slack : 2 * old_cap;
        }

        // Allocate a CSR with one slot per row, in row order, plus Slack's free
        // tail. Row r gets Slack::layout_capacity(len[r], grown[r]) slots and
        // keeps its grown flag; with no flags every row counts as never grown.
        // Row lengths start at zero.
        template <class Slack, class LayoutTag, class I, class V>
        csr_storage<LayoutTag, I, V> layout_rows(const std::vector<std::size_t> &len,
                                                 const std::vector<bool> *grown = nullptr)
        {
            const std::size_t n_rows = len.size();
            auto grown_at = [&](std::size_t r) { return grown != nullptr && (*grown)[r]; };

            std::size_t assigned = 0;
            for (std::size_t r = 0; r < n_rows; ++r)
                assigned += Slack::layout_capacity(len[r], grown_at(r));

            csr_storage<LayoutTag, I, V> csr(n_rows, assigned + Slack::tail(assigned));
            std::size_t pos = 0;
            for (std::size_t r = 0; r < n_rows; ++r)
            {
                csr.row_start[r] = pos;
                csr.row_len[r] = 0;
                csr.row_cap[r] = Slack::layout_capacity(len[r], grown_at(r));
                csr.row_grown[r] = grown_at(r);
                pos += csr.row_cap[r];
            }
            csr.end = assigned;
            return csr;
        }

        // Two-pointer merge of a row's committed entries (old_count of them,
        // read through old_col/old_val) with its sorted buffer, calling
        // emit(col, val) for each surviving entry in column order. The buffer
        // wins on a shared column; zeros (deletions) are dropped.
        template <class RowType, class OldCol, class OldVal, class Emit>
        void for_each_merged(const RowType &row, std::size_t old_count,
                             OldCol &&old_col, OldVal &&old_val, Emit &&emit)
        {
            using V = typename RowType::value_type;
            auto keep = [&](auto c, const V &v)
            {
                if (!traits::ValueTraits<V>::is_zero(v))
                    emit(c, v);
            };

            auto bit = row.begin();
            const auto bend = row.end();
            std::size_t oi = 0;
            while (oi < old_count && bit != bend)
            {
                const auto oc = old_col(oi);
                const auto bc = (*bit).first_ref();
                if (oc < bc)
                {
                    keep(oc, old_val(oi));
                    ++oi;
                }
                else
                {
                    keep(bc, (*bit).second_ref());
                    if (oc == bc)
                        ++oi;
                    ++bit;
                }
            }
            for (; oi < old_count; ++oi)
                keep(old_col(oi), old_val(oi));
            for (; bit != bend; ++bit)
                keep((*bit).first_ref(), (*bit).second_ref());
        }

        // Write the merge to consecutive slots from dst; returns the count.
        template <class LayoutTag, class I, class V, class RowType, class OldCol, class OldVal>
        std::size_t merge_row(csr_storage<LayoutTag, I, V> &out, std::size_t dst,
                              const RowType &row, std::size_t old_count,
                              OldCol &&old_col, OldVal &&old_val)
        {
            std::size_t wp = dst;
            for_each_merged(row, old_count, old_col, old_val,
                            [&](I c, const V &v) { out.set(wp++, c, v); });
            return wp - dst;
        }

        // Exact length row r will have once its buffer is merged into csr.
        template <class LayoutTag, class I, class V, class RowType>
        std::size_t merged_length(const csr_storage<LayoutTag, I, V> &csr, std::size_t r, const RowType &row)
        {
            const std::size_t start = csr.row_start[r];
            std::size_t n = 0;
            for_each_merged(row, csr.row_len[r],
                            [&](std::size_t k) { return csr.col(start + k); },
                            [&](std::size_t k) { return csr.val(start + k); },
                            [&](I, const V &) { ++n; });
            return n;
        }

        // Rebuild every row into a fresh allocation in row order: dirty rows
        // are merged with their buffers, clean rows are copied. Rows that have
        // ever grown past a slot keep slack (see config::row_slack). Used when
        // a re-lock would overflow the free tail or leave too many holes.
        template <class LayoutTag, class Slack, class RowType>
        auto repack(const csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type> &old,
                    const std::vector<RowType> &rows, const std::vector<bool> &is_dirty)
            -> csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type>
        {
            using I = typename RowType::index_type;
            using V = typename RowType::value_type;

            const std::size_t n_rows = rows.size();
            std::vector<std::size_t> len(n_rows);
            std::vector<bool> grown(n_rows);
            for (std::size_t r = 0; r < n_rows; ++r)
            {
                len[r] = is_dirty[r] ? merged_length(old, r, rows[r]) : old.row_len[r];
                grown[r] = old.row_grown[r] || len[r] > old.row_cap[r];
            }

            auto out = layout_rows<Slack, LayoutTag, I, V>(len, &grown);
            for (std::size_t r = 0; r < n_rows; ++r)
            {
                const std::size_t src = old.row_start[r];
                const std::size_t old_len = old.row_len[r];
                if (!is_dirty[r])
                {
                    out.copy_from(out.row_start[r], old, src, old_len);
                    out.row_len[r] = old_len;
                }
                else
                {
                    out.row_len[r] = merge_row(
                        out, out.row_start[r], rows[r], old_len,
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
    // No row has grown yet, so under slack_rows::edited every row is packed.
    // Precondition: every row in `rows` is locked.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class Slack = config::default_row_slack, class RowType>
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

        auto csr = detail::layout_rows<Slack, LayoutTag, I, V>(len);
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

    template <class LayoutTag, class Slack = config::default_row_slack, class RowType>
    bool relock_rows(csr_storage<LayoutTag, typename RowType::index_type, typename RowType::value_type> &csr,
                     std::vector<RowType> &rows, const std::vector<std::size_t> &dirty)
    {
        using I = typename RowType::index_type;
        using V = typename RowType::value_type;

        // Sort each dirty buffer and work out its exact merged length. Only a
        // row that really grows past its slot moves; total the tail space and
        // holes those moves would need.
        thread_local std::vector<std::size_t> new_len;
        new_len.resize(dirty.size());
        std::size_t tail_needed = 0;
        std::size_t freed = 0;
        for (std::size_t d = 0; d < dirty.size(); ++d)
        {
            const std::size_t r = dirty[d];
            rows[r].lock();
            new_len[d] = detail::merged_length(csr, r, rows[r]);
            if (new_len[d] > csr.row_cap[r])
            {
                tail_needed += detail::relocated_capacity<Slack>(new_len[d], csr.row_cap[r], csr.row_grown[r]);
                freed += csr.row_cap[r];
            }
        }

        const std::size_t end_after = csr.end + tail_needed;
        const bool out_of_tail = end_after > csr.capacity;
        const bool too_many_holes = Slack::should_repack(csr.holes + freed, end_after);

        if (out_of_tail || too_many_holes)
        {
            std::vector<bool> is_dirty(rows.size(), false);
            for (const std::size_t r : dirty)
                is_dirty[r] = true;
            csr = detail::repack<LayoutTag, Slack>(csr, rows, is_dirty);
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
        for (std::size_t d = 0; d < dirty.size(); ++d)
        {
            const std::size_t r = dirty[d];
            const std::size_t len = csr.row_len[r];
            const std::size_t start = csr.row_start[r];
            old_cols.resize(len);
            old_vals.resize(len);
            for (std::size_t k = 0; k < len; ++k)
            {
                old_cols[k] = csr.col(start + k);
                old_vals[k] = csr.val(start + k);
            }

            if (new_len[d] > csr.row_cap[r])
            {
                const std::size_t cap = detail::relocated_capacity<Slack>(new_len[d], csr.row_cap[r], csr.row_grown[r]);
                csr.holes += csr.row_cap[r];
                csr.row_start[r] = csr.end;
                csr.row_cap[r] = cap;
                csr.row_grown[r] = true;
                csr.end += cap;
            }

            detail::merge_row(csr, csr.row_start[r], rows[r], len,
                              [](std::size_t k) { return old_cols[k]; },
                              [](std::size_t k) { return old_vals[k]; });
            csr.nnz = csr.nnz - len + new_len[d];
            csr.row_len[r] = new_len[d];

            rows[r].set_csr_slice(csr.slice(r));
            rows[r].clear_buffer_content();
        }
        return false;
    }

} // namespace spira
