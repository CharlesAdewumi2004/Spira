#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/matrix/storage/csr_build.hpp>
#include <spira/matrix/storage/csr_storage.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/row.hpp>
#include <spira/traits.hpp>

namespace spira
{

    // ─────────────────────────────────────────────────────────────────────────────
    // matrix<LayoutTag, I, V, BufferTag, BufferN, Slack>
    //
    // Single-threaded dynamic sparse matrix with an open / locked lifecycle:
    //   open   — insert() / add() stage edits in per-row buffers.
    //   locked — reads and algorithms go through a flat CSR with per-row slack.
    // lock() commits only the rows edited since the previous lock; open() is O(1).
    // Slack (a config::row_slack) sets the CSR's spare room: per-row slack, the
    // free tail and the repack threshold. Arithmetic operators live in
    // matrix_operators.hpp.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, concepts::Indexable I = uint32_t,
              concepts::Valueable V = double,
              class BufferTag = buffer::tags::array_buffer<layout::tags::aos_tag>,
              std::size_t BufferN = 64,
              class Slack = config::default_row_slack>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V> &&
                 layout::ValidLayoutTag<LayoutTag> && config::is_row_slack_v<Slack>
    class matrix
    {
    public:
        using index_type = I;
        using value_type = V;
        using storage_type = row<LayoutTag, I, V, BufferTag, BufferN>;
        using slack_policy = Slack;
        using size_type = std::size_t;
        using shape_type = std::pair<size_type, size_type>;

        // ─────────────────────────────────────────
        // Construction / lifetime
        // ─────────────────────────────────────────

        matrix(size_type row_limit, size_type column_limit)
            : row_limit_{row_limit}, column_limit_{column_limit}, dirty_(row_limit, false)
        {
            rows_.reserve(row_limit_);
            for (size_type r = 0; r < row_limit_; ++r)
                rows_.emplace_back(column_limit_);
        }

        ~matrix() = default;

        // A copy re-points its rows at its own CSR; the defaulted copy would
        // leave them reading the source matrix's arrays.
        matrix(const matrix &other)
            : mode_{other.mode_}, rows_{other.rows_}, row_limit_{other.row_limit_},
              column_limit_{other.column_limit_}, csr_{other.csr_},
              dirty_{other.dirty_}, dirty_rows_{other.dirty_rows_}
        {
            if (csr_)
                install_slices<LayoutTag>(*csr_, rows_);
        }

        matrix &operator=(const matrix &other)
        {
            if (this != &other)
            {
                matrix tmp(other);
                *this = std::move(tmp);
            }
            return *this;
        }

        matrix(matrix &&) noexcept = default;
        matrix &operator=(matrix &&) noexcept = default;

        // ─────────────────────────────────────────
        // Shape and mode
        // ─────────────────────────────────────────

        [[nodiscard]] shape_type shape() const noexcept { return {row_limit_, column_limit_}; }
        [[nodiscard]] size_type n_rows() const noexcept { return row_limit_; }
        [[nodiscard]] size_type n_cols() const noexcept { return column_limit_; }

        [[nodiscard]] config::matrix_mode mode() const noexcept { return mode_; }
        [[nodiscard]] bool is_locked() const noexcept { return mode_ == config::matrix_mode::locked; }
        [[nodiscard]] bool is_open() const noexcept { return mode_ == config::matrix_mode::open; }

        /// Commit pending edits and freeze the matrix. Only rows edited since
        /// the last lock() are merged; the first lock() builds the whole CSR.
        void lock()
        {
            if (mode_ == config::matrix_mode::locked)
                return;

            if (!csr_)
            {
                // First lock: every row is laid out once.
                for (auto &r : rows_)
                    r.lock();
                csr_ = build_csr<LayoutTag, slack_policy>(rows_);
                install_slices<LayoutTag>(*csr_, rows_);
                for (const size_type r : dirty_rows_)
                    rows_[r].clear_buffer_content();
            }
            else
            {
                relock_rows<LayoutTag, slack_policy>(*csr_, rows_, dirty_rows_);
            }

            end_cycle();
            mode_ = config::matrix_mode::locked;
        }

        /// Transition back to mutable. O(1): rows stay locked until their first
        /// write (mark_dirty); a row with an empty buffer answers reads the same
        /// way in either mode.
        void open() { mode_ = config::matrix_mode::open; }

        // ─────────────────────────────────────────
        // Queries (both modes via row delegation)
        // ─────────────────────────────────────────

        /// Open mode: upper bound (committed + buffered). Locked mode: exact
        /// deduplicated count.
        [[nodiscard]] size_type row_nnz(index_type row_index) const { return row_at(row_index).size(); }

        [[nodiscard]] bool empty() const noexcept
        {
            for (const auto &r : rows_)
                if (!r.empty())
                    return false;
            return true;
        }

        [[nodiscard]] size_type nnz() const noexcept
        {
            size_type total = 0;
            for (const auto &r : rows_)
                total += r.size();
            return total;
        }

        [[nodiscard]] const storage_type &row_at(index_type row_index) const
        {
            validate_row_index(row_index);
            return rows_[to_size(row_index)];
        }

        /// Returns a pointer to the built CSR storage, or nullptr if not yet built.
        [[nodiscard]] const csr_storage<LayoutTag, I, V> *csr() const noexcept
        {
            return csr_ ? &*csr_ : nullptr;
        }

        [[nodiscard]] bool contains(index_type row_index, index_type col_index) const
        {
            validate_col_index(col_index);
            return row_at(row_index).contains(col_index);
        }

        [[nodiscard]] value_type get(index_type row_index, index_type col_index) const
        {
            validate_col_index(col_index);
            const auto *p = row_at(row_index).get(col_index);
            return p ? *p : traits::ValueTraits<value_type>::zero();
        }

        [[nodiscard]] value_type accumulate(index_type row_index) const
        {
            return row_at(row_index).accumulate();
        }

        // ─────────────────────────────────────────
        // Mutation (open mode)
        // ─────────────────────────────────────────

        void insert(index_type row_index, index_type col_index, const value_type &val)
        {
            writable_row(row_index, col_index, "insert").insert(col_index, val);
        }

        /// Add val onto the current value at (row, col); a missing entry counts
        /// as zero. A sum of exactly zero deletes the entry at the next lock().
        void add(index_type row_index, index_type col_index, const value_type &val)
        {
            writable_row(row_index, col_index, "add").add(col_index, val);
        }

        /// Drop every pending edit. Only queued rows can hold buffered entries;
        /// once emptied they are locked again so no row stays open across the
        /// next lock().
        void clear()
        {
            if (mode_ != config::matrix_mode::open)
                throw std::logic_error("matrix::clear() requires open mode");
            for (const size_type r : dirty_rows_)
            {
                rows_[r].clear();
                rows_[r].lock();
            }
            end_cycle();
        }

        /// Writable access to one row, queued for the next lock().
        [[nodiscard]] storage_type &row_at_mut(index_type row_index)
        {
            if (mode_ != config::matrix_mode::open)
                throw std::logic_error("matrix::row_at_mut() requires open mode");
            validate_row_index(row_index);
            mark_dirty(to_size(row_index));
            return rows_[to_size(row_index)];
        }

        // ─────────────────────────────────────────
        // Mode-independent
        // ─────────────────────────────────────────

        void swap(matrix &other) noexcept
        {
            using std::swap;
            swap(mode_, other.mode_);
            swap(rows_, other.rows_);
            swap(row_limit_, other.row_limit_);
            swap(column_limit_, other.column_limit_);
            swap(csr_, other.csr_);
            swap(dirty_, other.dirty_);
            swap(dirty_rows_, other.dirty_rows_);
        }

        /// Directly install a pre-built, sorted CSR into the matrix and transition
        /// to locked mode, discarding any pending edits. Validates that the CSR is
        /// well-formed: row count matches, every row slot lies inside the assigned
        /// slots, row lengths sum to nnz, every row's column indices are strictly
        /// sorted (ascending, no duplicates), and all column indices are in bounds.
        /// Slots are assumed not to overlap. Throws std::invalid_argument if any
        /// check fails.
        void load_csr(csr_storage<LayoutTag, I, V> &&csr)
        {
            auto fail = [](const char *what)
            { throw std::invalid_argument(std::string("spira::matrix::load_csr: ") + what); };

            if (csr.n_rows != rows_.size())
                fail("CSR row count does not match matrix dimensions");
            if (!csr.is_built())
                fail("CSR row table is null");
            if (csr.end > csr.capacity)
                fail("CSR assigned slots exceed its capacity");

            std::size_t total = 0;
            for (std::size_t i = 0; i < csr.n_rows; ++i)
            {
                const std::size_t beg = csr.row_start[i];
                const std::size_t len = csr.row_len[i];
                if (len > csr.row_cap[i] || beg + csr.row_cap[i] > csr.end)
                    fail("CSR row slot lies outside the assigned slots");
                total += len;

                for (std::size_t k = beg; k < beg + len; ++k)
                {
                    const I col = csr.col(k);
                    if (static_cast<size_type>(col) >= column_limit_)
                        fail("column index out of bounds");
                    if (k > beg && col <= csr.col(k - 1))
                        fail("CSR row entries are not strictly sorted by column");
                }
            }
            if (total != csr.nnz)
                fail("CSR row lengths do not sum to nnz");

            for (auto &r : rows_)
            {
                r.clear_buffer_content();
                r.lock();
            }
            end_cycle();

            csr_ = std::move(csr);
            mode_ = config::matrix_mode::locked;
            install_slices<LayoutTag>(*csr_, rows_);
        }

        // ─────────────────────────────────────────
        // Iteration
        // ─────────────────────────────────────────

        /// Calls f(row, row_index) for every row.
        template <class Func>
        void for_each_row(Func &&f) const
        {
            for (size_type i = 0; i < row_limit_; ++i)
                f(rows_[i], static_cast<index_type>(i));
        }

        /// Calls f(row, row_index) for every non-empty row.
        template <class Func>
        void for_each_nnz_row(Func &&f) const
        {
            for (size_type i = 0; i < row_limit_; ++i)
                if (!rows_[i].empty())
                    f(rows_[i], static_cast<index_type>(i));
        }

    private:
        static constexpr size_type to_size(index_type i) noexcept
        {
            return static_cast<size_type>(i);
        }

        void validate_row_index(index_type row_index) const
        {
            if (to_size(row_index) >= row_limit_)
                throw std::out_of_range("spira::matrix: row_index out of range");
        }

        void validate_col_index(index_type col_index) const
        {
            if (to_size(col_index) >= column_limit_)
                throw std::out_of_range("spira::matrix: col_index out of range");
        }

        /// The row an edit at (row, col) goes to, checked and queued.
        storage_type &writable_row(index_type row_index, index_type col_index, const char *op)
        {
            if (mode_ != config::matrix_mode::open)
                throw std::logic_error(std::string("matrix::") + op + "() requires open mode");
            validate_row_index(row_index);
            validate_col_index(col_index);
            mark_dirty(to_size(row_index));
            return rows_[to_size(row_index)];
        }

        /// Record that row r has pending edits: reopen it and queue it for the
        /// next lock(). Each row is queued at most once per cycle.
        void mark_dirty(size_type r)
        {
            if (dirty_[r])
                return;
            dirty_[r] = true;
            dirty_rows_.push_back(r);
            rows_[r].open();
        }

        /// Forget the rows queued in this cycle (after a lock, clear or load).
        void end_cycle()
        {
            for (const size_type r : dirty_rows_)
                dirty_[r] = false;
            dirty_rows_.clear();
        }

        config::matrix_mode mode_{config::matrix_mode::open};
        std::vector<storage_type> rows_{};
        size_type row_limit_{0};
        size_type column_limit_{0};
        std::optional<csr_storage<LayoutTag, I, V>> csr_{};
        std::vector<bool> dirty_{};           // row r is in dirty_rows_
        std::vector<size_type> dirty_rows_{}; // rows edited since the last lock()
    };

} // namespace spira
