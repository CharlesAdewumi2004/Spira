#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spira/config.hpp>
#include <spira/concepts.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/matrix/buffer/buffer_tags.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/storage/csr_build.hpp>
#include <spira/matrix/storage/csr_storage.hpp>
#include <spira/matrix/row.hpp>
#include <spira/traits.hpp>
#include <spira/parallel/insert_staging.hpp>
#include <spira/parallel/partition.hpp>
#include <spira/parallel/thread_pool.hpp>

namespace spira::parallel
{

    // ─────────────────────────────────────────────────────────────────────────────
    // parallel_matrix<LayoutTag, I, V, BufferTag, BufferN, LP, IP, StagingN>
    //
    // Sparse matrix whose rows are statically partitioned across n_threads worker
    // threads.  The public API mirrors spira::matrix for the core operations:
    //   insert()  — routes to the owning partition; no thread involvement.
    //   lock()    — parallel: each worker locks its own partition independently.
    //   open()    — parallel: each worker reopens its own partition rows.
    //
    // Row ownership is determined by integer-division routing:
    //   owner(row) = row * n_threads / n_rows   (uniform row-count split)
    //
    // Each partition owns its rows and its own csr_storage; there is no shared
    // flat CSR across partitions.  partition_at(t) gives read access to any
    // partition for algorithms that iterate over the full matrix.
    //
    // Insert policy (IP):
    //   direct  — main thread writes straight to partition row buffers.
    //             Zero overhead, but cache-hostile under random row arrival order.
    //   staged  — main thread accumulates into StagingN-entry per-partition arrays,
    //             burst-flushing to row buffers when full or at lock().
    //             Keeps the hot staging array in L1/L2, reduces random-access misses.
    //
    // Not copyable (owns a thread_pool). Moveable only if the pool is idle.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag,
              concepts::Indexable I = uint32_t,
              concepts::Valueable V = double,
              class BufferTag = buffer::tags::array_buffer<layout::tags::aos_tag>,
              std::size_t BufferN = 64,
              config::lock_policy   LP       = config::lock_policy::compact_preserve,
              config::insert_policy IP       = config::insert_policy::direct,
              std::size_t           StagingN = 256>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V> &&
                 layout::ValidLayoutTag<LayoutTag>
    class parallel_matrix
    {
    public:
        using partition_type = partition<LayoutTag, I, V, BufferTag, BufferN, LP>;
        using row_type = typename partition_type::row_type;
        using index_type = I;
        using value_type = V;
        using size_type = std::size_t;
        using shape_type = std::pair<size_type, size_type>;

        // ─────────────────────────────────────────
        // Construction
        // ─────────────────────────────────────────

        parallel_matrix(size_type n_rows, size_type n_cols, size_type n_threads);
        parallel_matrix(size_type n_rows, size_type n_cols, size_type n_threads,
                        size_type reserve_per_row);

        ~parallel_matrix() = default;

        // Not copyable — thread_pool is not copyable.
        parallel_matrix(const parallel_matrix &) = delete;
        parallel_matrix &operator=(const parallel_matrix &) = delete;

        // Moveable — pool must be idle at time of move.
        parallel_matrix(parallel_matrix &&) = default;
        parallel_matrix &operator=(parallel_matrix &&) = default;

        // ─────────────────────────────────────────
        // Shape
        // ─────────────────────────────────────────

        [[nodiscard]] size_type n_rows() const noexcept { return n_rows_; }
        [[nodiscard]] size_type n_cols() const noexcept { return n_cols_; }
        [[nodiscard]] size_type n_threads() const noexcept { return pool_.size(); }
        [[nodiscard]] shape_type shape() const noexcept { return {n_rows_, n_cols_}; }

        // ─────────────────────────────────────────
        // Mode
        // ─────────────────────────────────────────

        [[nodiscard]] config::matrix_mode mode() const noexcept { return mode_; }
        [[nodiscard]] bool is_locked() const noexcept
        {
            return mode_ == config::matrix_mode::locked;
        }
        [[nodiscard]] bool is_open() const noexcept
        {
            return mode_ == config::matrix_mode::open;
        }

        /// Parallel lock: each worker thread sorts, deduplicates, and builds the
        /// CSR for its own partition.  Blocks until all partitions are locked.
        void lock();

        /// Parallel open: each worker thread reopens all rows in its partition.
        /// Blocks until all partitions are open.  The per-partition CSR is kept
        /// (used as the base for merge_csr on the next lock cycle).
        void open();

        // ─────────────────────────────────────────
        // Queries (both modes)
        // ─────────────────────────────────────────

        /// Total non-zeros across all partitions.
        /// Open mode: upper bound. Locked mode: exact deduplicated count.
        [[nodiscard]] size_type nnz() const noexcept;

        [[nodiscard]] bool empty() const noexcept;

        /// NNZ for a single global row.
        [[nodiscard]] size_type row_nnz(size_type row_idx) const;

        /// Read-only access to a global row (routes to owning partition).
        [[nodiscard]] const row_type &row_at(size_type row_idx) const;

        [[nodiscard]] bool contains(size_type row_idx, I col_idx) const;

        /// Returns the stored value at (row, col), or zero if absent.
        [[nodiscard]] value_type get(size_type row_idx, I col_idx) const;

        /// Sum of all values in a row (last-write-wins dedup in open mode).
        [[nodiscard]] value_type accumulate(size_type row_idx) const;

        // ─────────────────────────────────────────
        // Mutation (open mode)
        // ─────────────────────────────────────────

        /// Routes (row, col, val) to the owning partition and inserts into the
        /// corresponding row buffer.  Called from the user's thread; no pool
        /// involvement.
        void insert(size_type row_idx, I col, V val);

        /// Clear all row buffers across every partition.
        void clear() noexcept;

        // ─────────────────────────────────────────
        // Iteration
        // ─────────────────────────────────────────

        /// Calls f(row, global_row_index) for every row in global order.
        template <class Func>
        void for_each_row(Func &&f) const;

        template <class Func>
        void for_each_row(Func &&f);

        /// Calls f(row, global_row_index) only for non-empty rows.
        template <class Func>
        void for_each_nnz_row(Func &&f) const;

        // ─────────────────────────────────────────
        // Partition access
        // ─────────────────────────────────────────

        [[nodiscard]] const partition_type &partition_at(size_type t) const
        {
            assert(t < parts_.size());
            return parts_[t];
        }

        [[nodiscard]] partition_type &partition_at(size_type t)
        {
            assert(t < parts_.size());
            return parts_[t];
        }

    private:
        // Determine which partition owns global_row.
        [[nodiscard]] size_type owner(size_type global_row) const noexcept
        {
            return global_row * parts_.size() / n_rows_;
        }

        void validate_row(size_type r) const
        {
            if (r >= n_rows_)
                throw std::out_of_range("parallel_matrix: row index out of range");
        }

        void validate_col(size_type c) const
        {
            if (c >= n_cols_)
                throw std::out_of_range("parallel_matrix: col index out of range");
        }

        // Lock a single partition (called inside the pool execute lambda).
        void lock_partition(partition_type &p);

    private:
        size_type n_rows_;
        size_type n_cols_;
        config::matrix_mode mode_{config::matrix_mode::open};
        std::vector<partition_type> parts_;
        thread_pool pool_;
        // Zero-size for insert_policy::direct (empty specialisation).
        [[no_unique_address]] insert_staging<IP, I, V, StagingN> staging_;
    };

    // ═════════════════════════════════════════════════════════════════════════════
    // Construction
    // ═════════════════════════════════════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::parallel_matrix(
        size_type n_rows, size_type n_cols, size_type n_threads)
        : parallel_matrix(n_rows, n_cols, n_threads, config::default_row_reserve_hint)
    {
    }

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::parallel_matrix(
        size_type n_rows, size_type n_cols, size_type n_threads, size_type reserve_per_row)
        : n_rows_{n_rows}, n_cols_{n_cols}, pool_{n_threads}
    {
        assert(n_threads >= 1 && "parallel_matrix requires at least one thread");

        parts_.resize(n_threads);
        for (size_type t = 0; t < n_threads; ++t)
        {
            auto &p = parts_[t];
            p.row_start = t * n_rows / n_threads;
            p.row_end = (t + 1) * n_rows / n_threads;
            const size_type local_n = p.row_end - p.row_start;
            p.rows.reserve(local_n);
            for (size_type r = 0; r < local_n; ++r)
                p.rows.emplace_back(reserve_per_row, n_cols);
        }

        if constexpr (IP == config::insert_policy::staged)
            staging_.init(n_threads);
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // Mode transitions
    // ═════════════════════════════════════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::lock_partition(partition_type &p)
    {
        // Detach stale CSR slice pointers before rebuilding.
        for (auto &r : p.rows)
            r.reset_csr_slice();

        // Sort + dedup + filter each row's buffer in-place.
        for (auto &r : p.rows)
            r.lock();

        if constexpr (LP == config::lock_policy::compact_preserve ||
                      LP == config::lock_policy::compact_move)
        {
            // Build or merge the flat CSR for this partition.
            // offsets == nullptr signals "first lock cycle".
            if (p.csr.offsets != nullptr)
                p.csr = merge_csr<L>(p.rows, std::move(p.csr));
            else
                p.csr = build_csr<L>(p.rows);

            // Install layout-appropriate CSR slices on every row.
            const std::size_t *off = p.csr.offsets.get();
            if constexpr (std::is_same_v<L, layout::tags::soa_tag>)
            {
                const I *cols_flat = p.csr.cols.get();
                const V *vals_flat = p.csr.vals.get();
                for (std::size_t i = 0; i < p.rows.size(); ++i)
                    p.rows[i].set_csr_slice(csr_slice<L, I, V>{
                        cols_flat + off[i], vals_flat + off[i], off[i + 1] - off[i]});
            }
            else // aos_tag
            {
                const auto *pairs_flat = p.csr.pairs.get();
                for (std::size_t i = 0; i < p.rows.size(); ++i)
                    p.rows[i].set_csr_slice(csr_slice<L, I, V>{
                        pairs_flat + off[i], off[i + 1] - off[i]});
            }

            // Clear staging buffers — data now lives in the flat CSR.
            for (auto &r : p.rows)
                r.clear_buffer_content();

            if constexpr (LP == config::lock_policy::compact_move)
                for (auto &r : p.rows)
                    r.release_buffer();
        }
    }

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::lock()
    {
        if (mode_ == config::matrix_mode::locked)
            return;

        // Flush any staged inserts into row buffers before workers lock.
        if constexpr (IP == config::insert_policy::staged)
            staging_.flush_all(parts_);

        pool_.execute([this](std::size_t t) { lock_partition(parts_[t]); });

        mode_ = config::matrix_mode::locked;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::open()
    {
        if (mode_ == config::matrix_mode::open)
            return;

        pool_.execute([this](std::size_t t)
                      {
            for (auto &r : parts_[t].rows)
                r.open(); });

        mode_ = config::matrix_mode::open;
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // Queries
    // ═════════════════════════════════════════════════════════════════════════════

#define SPIRA_PM_TMPL \
    template <class L, concepts::Indexable I, concepts::Valueable V, \
              class BT, std::size_t BN, config::lock_policy LP, \
              config::insert_policy IP, std::size_t SN> \
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && \
                 layout::ValidLayoutTag<L>

    SPIRA_PM_TMPL
    auto parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::nnz() const noexcept -> size_type
    {
        size_type total = 0;
        for (const auto &p : parts_)
            for (const auto &r : p.rows)
                total += r.size();
        return total;
    }

    SPIRA_PM_TMPL
    bool parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::empty() const noexcept
    {
        for (const auto &p : parts_)
            for (const auto &r : p.rows)
                if (!r.empty())
                    return false;
        return true;
    }

    SPIRA_PM_TMPL
    auto parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::row_nnz(size_type row_idx) const -> size_type
    {
        validate_row(row_idx);
        const auto &p = parts_[owner(row_idx)];
        return p.rows[p.local_row(row_idx)].size();
    }

    SPIRA_PM_TMPL
    auto parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::row_at(size_type row_idx) const
        -> const row_type &
    {
        validate_row(row_idx);
        const auto &p = parts_[owner(row_idx)];
        return p.rows[p.local_row(row_idx)];
    }

    SPIRA_PM_TMPL
    bool parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::contains(size_type row_idx, I col_idx) const
    {
        validate_row(row_idx);
        validate_col(static_cast<size_type>(col_idx));
        const auto &p = parts_[owner(row_idx)];
        return p.rows[p.local_row(row_idx)].contains(col_idx);
    }

    SPIRA_PM_TMPL
    auto parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::get(size_type row_idx, I col_idx) const
        -> value_type
    {
        validate_row(row_idx);
        validate_col(static_cast<size_type>(col_idx));
        const auto &p   = parts_[owner(row_idx)];
        const auto *ptr = p.rows[p.local_row(row_idx)].get(col_idx);
        return ptr ? *ptr : traits::ValueTraits<value_type>::zero();
    }

    SPIRA_PM_TMPL
    auto parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::accumulate(size_type row_idx) const
        -> value_type
    {
        validate_row(row_idx);
        const auto &p = parts_[owner(row_idx)];
        return p.rows[p.local_row(row_idx)].accumulate();
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // Mutation
    // ═════════════════════════════════════════════════════════════════════════════

    SPIRA_PM_TMPL
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::insert(size_type row_idx, I col, V val)
    {
        assert(mode_ == config::matrix_mode::open &&
               "parallel_matrix::insert() requires open mode");
        validate_row(row_idx);
        validate_col(static_cast<size_type>(col));

        const size_type t    = owner(row_idx);
        auto           &p   = parts_[t];
        const size_type loc = p.local_row(row_idx);

        if constexpr (IP == config::insert_policy::staged)
        {
            staging_.bufs_[t].push_back({loc, col, val});
            if (staging_.bufs_[t].size() >= SN)
                staging_.flush(t, p);
        }
        else
        {
            p.rows[loc].insert(col, val);
        }
    }

    SPIRA_PM_TMPL
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::clear() noexcept
    {
        assert(mode_ == config::matrix_mode::open &&
               "parallel_matrix::clear() requires open mode");
        for (auto &p : parts_)
            for (auto &r : p.rows)
                r.clear();
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // Iteration
    // ═════════════════════════════════════════════════════════════════════════════

    SPIRA_PM_TMPL
    template <class Func>
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::for_each_row(Func &&f) const
    {
        for (const auto &p : parts_)
            for (size_type i = 0; i < p.rows.size(); ++i)
                f(p.rows[i], static_cast<index_type>(p.row_start + i));
    }

    SPIRA_PM_TMPL
    template <class Func>
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::for_each_row(Func &&f)
    {
        for (auto &p : parts_)
            for (size_type i = 0; i < p.rows.size(); ++i)
                f(p.rows[i], static_cast<index_type>(p.row_start + i));
    }

    SPIRA_PM_TMPL
    template <class Func>
    void parallel_matrix<L, I, V, BT, BN, LP, IP, SN>::for_each_nnz_row(Func &&f) const
    {
        for (const auto &p : parts_)
            for (size_type i = 0; i < p.rows.size(); ++i)
                if (!p.rows[i].empty())
                    f(p.rows[i], static_cast<index_type>(p.row_start + i));
    }

#undef SPIRA_PM_TMPL

} // namespace spira::parallel
