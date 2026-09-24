#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
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
    // parallel_matrix<LayoutTag, I, V, BufferTag, BufferN, IP, StagingN, Slack>
    //
    // Sparse matrix whose rows are split into contiguous partitions, one per
    // worker thread. The public API mirrors spira::matrix for the core operations:
    //   insert() / add() — route to the owning partition; no thread involvement.
    //   lock()           — parallel: each worker commits its own partition's
    //                      edited rows.
    //   open()           — O(1); rows reopen lazily on their first write.
    //
    // Rows are split evenly by count: partition t owns rows
    // [t * n_rows / n_threads, (t + 1) * n_rows / n_threads).
    //
    // Each partition owns its rows and its own csr_storage; there is no shared
    // flat CSR across partitions. partition_at(t) gives access to any partition
    // for algorithms that iterate over the full matrix.
    //
    // Insert policy (IP): see insert_staging.hpp.
    // Slack: the partitions' CSR spare room (see config::row_slack).
    //
    // Not copyable (owns a thread_pool). Moveable only if the pool is idle.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag,
              concepts::Indexable I = uint32_t,
              concepts::Valueable V = double,
              class BufferTag = buffer::tags::array_buffer<layout::tags::aos_tag>,
              std::size_t BufferN = 64,
              config::insert_policy IP       = config::insert_policy::direct,
              std::size_t           StagingN = 256,
              class                 Slack    = config::default_row_slack>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V> &&
                 layout::ValidLayoutTag<LayoutTag> && config::is_row_slack_v<Slack>
    class parallel_matrix
    {
    public:
        using partition_type = partition<LayoutTag, I, V, BufferTag, BufferN>;
        using row_type = typename partition_type::row_type;
        using index_type = I;
        using value_type = V;
        using size_type = std::size_t;
        using shape_type = std::pair<size_type, size_type>;
        using slack_policy = Slack;

        // ─────────────────────────────────────────
        // Construction
        // ─────────────────────────────────────────

        /// Throws std::invalid_argument if n_threads is 0.
        parallel_matrix(size_type n_rows, size_type n_cols, size_type n_threads)
            : n_rows_{n_rows}, n_cols_{n_cols}, pool_{std::make_unique<thread_pool>(n_threads)}
        {
            parts_.resize(n_threads);
            for (size_type t = 0; t < n_threads; ++t)
            {
                auto &p = parts_[t];
                p.row_start = t * n_rows / n_threads;
                p.row_end = (t + 1) * n_rows / n_threads;
                p.rows.reserve(p.size());
                for (size_type r = 0; r < p.size(); ++r)
                    p.rows.emplace_back(n_cols);
                p.reset_dirty(p.size());
            }
            if constexpr (staged)
                staging_.init(n_threads);
        }

        ~parallel_matrix() = default;

        parallel_matrix(const parallel_matrix &) = delete;
        parallel_matrix &operator=(const parallel_matrix &) = delete;

        // Moveable — pool must be idle at time of move.
        parallel_matrix(parallel_matrix &&) = default;
        parallel_matrix &operator=(parallel_matrix &&) = default;

        // ─────────────────────────────────────────
        // Shape and mode
        // ─────────────────────────────────────────

        [[nodiscard]] size_type n_rows() const noexcept { return n_rows_; }
        [[nodiscard]] size_type n_cols() const noexcept { return n_cols_; }
        [[nodiscard]] size_type n_threads() const noexcept { return pool_->size(); }
        [[nodiscard]] shape_type shape() const noexcept { return {n_rows_, n_cols_}; }

        [[nodiscard]] config::matrix_mode mode() const noexcept { return mode_; }
        [[nodiscard]] bool is_locked() const noexcept { return mode_ == config::matrix_mode::locked; }
        [[nodiscard]] bool is_open() const noexcept { return mode_ == config::matrix_mode::open; }

        /// Parallel lock: each worker thread commits the rows of its own
        /// partition that were edited since the last lock (the first lock
        /// builds every row). Blocks until all partitions are locked.
        void lock()
        {
            if (mode_ == config::matrix_mode::locked)
                return;
            if constexpr (staged)
                staging_.flush_all(parts_);
            pool_->execute([this](std::size_t t) { lock_partition(parts_[t]); });
            mode_ = config::matrix_mode::locked;
        }

        /// O(1). Rows are reopened lazily on their first write; the
        /// per-partition CSR is kept as the base for the next lock.
        void open() { mode_ = config::matrix_mode::open; }

        // ─────────────────────────────────────────
        // Queries (both modes)
        // ─────────────────────────────────────────

        /// Total non-zeros across all partitions.
        /// Open mode: upper bound. Locked mode: exact deduplicated count.
        [[nodiscard]] size_type nnz() const noexcept
        {
            size_type total = 0;
            for (const auto &p : parts_)
                for (const auto &r : p.rows)
                    total += r.size();
            return total;
        }

        [[nodiscard]] bool empty() const noexcept
        {
            for (const auto &p : parts_)
                for (const auto &r : p.rows)
                    if (!r.empty())
                        return false;
            return true;
        }

        /// NNZ for a single global row.
        [[nodiscard]] size_type row_nnz(size_type row_idx) const { return row_at(row_idx).size(); }

        /// Read-only access to a global row (routes to owning partition).
        [[nodiscard]] const row_type &row_at(size_type row_idx) const
        {
            validate_row(row_idx);
            const auto &p = parts_[owner(row_idx)];
            return p.rows[p.local_row(row_idx)];
        }

        [[nodiscard]] bool contains(size_type row_idx, I col_idx) const
        {
            validate_col(static_cast<size_type>(col_idx));
            return row_at(row_idx).contains(col_idx);
        }

        /// Returns the stored value at (row, col), or zero if absent.
        [[nodiscard]] value_type get(size_type row_idx, I col_idx) const
        {
            validate_col(static_cast<size_type>(col_idx));
            const auto *ptr = row_at(row_idx).get(col_idx);
            return ptr ? *ptr : traits::ValueTraits<value_type>::zero();
        }

        /// Sum of all values in a row (last-write-wins dedup in open mode).
        [[nodiscard]] value_type accumulate(size_type row_idx) const
        {
            return row_at(row_idx).accumulate();
        }

        // ─────────────────────────────────────────
        // Mutation (open mode)
        // ─────────────────────────────────────────

        /// Routes (row, col, val) to the owning partition and inserts into the
        /// corresponding row buffer. Called from the user's thread; no pool
        /// involvement.
        void insert(size_type row_idx, I col, V val) { route(row_idx, col, val, false, "insert"); }

        /// Like insert(), but adds val onto the current value at (row, col);
        /// a missing entry counts as zero. A sum of exactly zero deletes the
        /// entry at the next lock(). Under the staged policy the delta is
        /// summed when the staging array is flushed, so it also builds on
        /// inserts that are still staged.
        void add(size_type row_idx, I col, V val) { route(row_idx, col, val, true, "add"); }

        /// Drop every pending edit, staged or buffered.
        void clear()
        {
            if (mode_ != config::matrix_mode::open)
                throw std::logic_error("parallel_matrix::clear() requires open mode");
            if constexpr (staged)
                staging_.clear();
            for (auto &p : parts_)
                p.clear_pending();
        }

        /// Parallel bulk fill — for batch assembly when source data can be
        /// partitioned by row range ahead of time.
        ///
        /// f is invoked once per worker thread as:
        ///   f(rows, row_start, row_end, thread_id)
        ///
        /// where rows is the std::vector<row_type> for that partition and
        /// [row_start, row_end) is the global row range it owns. Insert via:
        ///   rows[global_row - row_start].insert(col, val)
        ///
        /// Matrix must be open. Stays open after parallel_fill — call lock()
        /// when done. No routing overhead; each thread writes only to its own
        /// partition.
        template <class Func>
        void parallel_fill(Func &&f)
        {
            if (mode_ != config::matrix_mode::open)
                throw std::logic_error("parallel_fill: matrix must be open");
            pool_->execute([this, &f](std::size_t t)
            {
                auto &p = parts_[t];
                p.open_all();
                f(p.rows, p.row_start, p.row_end, t);
            });
        }

        // ─────────────────────────────────────────
        // Iteration
        // ─────────────────────────────────────────

        /// Calls f(row, global_row_index) for every row in global order.
        template <class Func>
        void for_each_row(Func &&f) const
        {
            for (const auto &p : parts_)
                for (size_type i = 0; i < p.rows.size(); ++i)
                    f(p.rows[i], static_cast<index_type>(p.row_start + i));
        }

        // ─────────────────────────────────────────
        // Partition access
        // ─────────────────────────────────────────

        [[nodiscard]] const partition_type &partition_at(size_type t) const
        {
            if (t >= parts_.size())
                throw std::out_of_range("parallel_matrix::partition_at: thread index out of range");
            return parts_[t];
        }

        [[nodiscard]] partition_type &partition_at(size_type t)
        {
            if (t >= parts_.size())
                throw std::out_of_range("parallel_matrix::partition_at: thread index out of range");
            return parts_[t];
        }

        /// Run f(partition, thread_id) on every partition in parallel.
        /// Blocks until all workers have returned from f.
        /// Used by free-function algorithms that need parallel partition access.
        /// Writes into a partition's rows must go through writable_row().
        template <class Func>
        void execute(Func &&f)
        {
            pool_->execute([this, &f](std::size_t t) { f(parts_[t], t); });
        }

    private:
        static constexpr bool staged = IP == config::insert_policy::staged;

        // Partition owning global_row: a short linear scan over the stored
        // boundaries (n_threads is small), which avoids re-deriving the
        // integer-division split.
        [[nodiscard]] size_type owner(size_type global_row) const noexcept
        {
            for (size_type t = 0; t + 1 < parts_.size(); ++t)
                if (global_row < parts_[t + 1].row_start)
                    return t;
            return parts_.size() - 1;
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

        void route(size_type row_idx, I col, V val, bool add, const char *op)
        {
            if (mode_ != config::matrix_mode::open)
                throw std::logic_error(std::string("parallel_matrix::") + op + "() requires open mode");
            validate_row(row_idx);
            validate_col(static_cast<size_type>(col));

            const size_type t = owner(row_idx);
            auto &p = parts_[t];
            const size_type loc = p.local_row(row_idx);

            if constexpr (staged)
            {
                staging_.push(t, p, {loc, col, add, val});
            }
            else
            {
                auto &row = p.writable_row(loc);
                if (add)
                    row.add(col, val);
                else
                    row.insert(col, val);
            }
        }

        // Lock a single partition (called inside the pool execute lambda).
        void lock_partition(partition_type &p)
        {
            if (!p.csr.is_built())
            {
                // First lock: every row is laid out once.
                for (auto &r : p.rows)
                    r.lock();
                p.csr = build_csr<LayoutTag, slack_policy>(p.rows);
                install_slices<LayoutTag>(p.csr, p.rows);
                for (auto &r : p.rows)
                    r.clear_buffer_content();
            }
            else
            {
                if (p.scan_all)
                {
                    // Rows were handed out wholesale (parallel_fill):
                    // any row with a buffered entry is dirty, the rest go back to locked.
                    for (std::size_t i = 0; i < p.rows.size(); ++i)
                    {
                        if (p.rows[i].has_buffered())
                            p.mark_dirty(i);
                        else
                            p.rows[i].lock();
                    }
                }
                relock_rows<LayoutTag, slack_policy>(p.csr, p.rows, p.dirty_rows);
            }
            p.end_cycle();
        }

        size_type n_rows_;
        size_type n_cols_;
        config::matrix_mode mode_{config::matrix_mode::open};
        std::vector<partition_type> parts_;
        std::unique_ptr<thread_pool> pool_;
        [[no_unique_address]] insert_staging<IP, I, V, StagingN> staging_;
    };

} // namespace spira::parallel
