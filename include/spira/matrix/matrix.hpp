#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/matrix/csr_build.hpp>
#include <spira/matrix/csr_storage.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/row.hpp>
#include <spira/traits.hpp>

namespace spira
{

    template <class LayoutTag, concepts::Indexable I = uint32_t,
              concepts::Valueable V = double,
              class BufferTag = buffer::tags::array_buffer<layout::tags::aos_tag>,
              std::size_t BufferN = 64,
              config::lock_policy LP = config::lock_policy::compact_preserve>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V> &&
                 layout::ValidLayoutTag<LayoutTag>
    class matrix
    {
    public:
        using index_type = I;
        using value_type = V;
        using storage_type = row<LayoutTag, I, V, BufferTag, BufferN>;
        using size_type = std::size_t;
        using shape_type = std::pair<size_type, size_type>;

        // ─────────────────────────────────────────
        // Construction / lifetime
        // ─────────────────────────────────────────

        explicit matrix(size_type row_limit, size_type column_limit);
        matrix(size_type row_limit, size_type column_limit,
               size_type reserve_per_row);

        ~matrix() = default;
        matrix(const matrix &) = default;
        matrix(matrix &&) noexcept = default;
        matrix &operator=(const matrix &) = default;
        matrix &operator=(matrix &&) noexcept = default;

        // ─────────────────────────────────────────
        // Shape
        // ─────────────────────────────────────────

        [[nodiscard]] shape_type shape() const noexcept;
        [[nodiscard]] size_type n_rows() const noexcept;
        [[nodiscard]] size_type n_cols() const noexcept;

        // ─────────────────────────────────────────
        // Mode transitions
        // ─────────────────────────────────────────

        [[nodiscard]] config::matrix_mode mode() const noexcept;
        [[nodiscard]] bool is_locked() const noexcept;
        [[nodiscard]] bool is_open() const noexcept;

        /// Sort + deduplicate every row, freeze the matrix.
        /// After this, no mutations allowed, zero-overhead reads.
        void lock();

        /// Transition back to mutable. Slab preserved; buffer ready for new inserts.
        void open();

        // ─────────────────────────────────────────
        // Queries (both modes via row delegation)
        // ─────────────────────────────────────────

        /// Open mode: upper bound (slab + buffer). Locked mode: exact deduplicated
        /// count.
        [[nodiscard]] size_type row_nnz(index_type row_index) const;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_type nnz() const noexcept;

        [[nodiscard]] const storage_type &row_at(index_type row_index) const;

        /// Returns a pointer to the built CSR storage, or nullptr if not yet built.
        [[nodiscard]] const csr_storage<LayoutTag, I, V> *csr() const noexcept
        {
            return csr_ ? &*csr_ : nullptr;
        }

        [[nodiscard]] bool contains(index_type row_index, index_type col_index) const;
        [[nodiscard]] value_type get(index_type row_index,
                                     index_type col_index) const;
        [[nodiscard]] value_type accumulate(index_type row_index) const;

        // ─────────────────────────────────────────
        // Mutation (open mode)
        // ─────────────────────────────────────────

        void insert(index_type row_index, index_type col_index,
                    const value_type &val);
        void clear() noexcept;

        [[nodiscard]] storage_type &row_at_mut(index_type row_index);

        // ─────────────────────────────────────────
        // Mode-independent
        // ─────────────────────────────────────────

        void swap(matrix &other) noexcept;

        // ─────────────────────────────────────────
        // Iteration helpers
        // ─────────────────────────────────────────

        template <class Func>
        void for_each_row(Func &&f) const;

        template <class Func>
        void for_each_row(Func &&f);

        template <class Func>
        void for_each_nnz_row(Func &&f) const;

        // ─────────────────────────────────────────
        // Arithmetic (locked mode — returns new matrices)
        // ─────────────────────────────────────────

        matrix operator+(const matrix &other) const;
        matrix operator-(const matrix &other) const;
        matrix &operator+=(const matrix &other);
        matrix &operator-=(const matrix &other);

        matrix operator*(const matrix &other) const;
        matrix &operator*=(const matrix &other);

        std::vector<value_type> operator*(const std::vector<value_type> &x) const;

        matrix operator*(value_type s) const;
        matrix &operator*=(value_type s);
        matrix operator/(value_type s) const;
        matrix &operator/=(value_type s);

        matrix operator~() const;

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

    private:
        config::matrix_mode mode_{config::matrix_mode::open};
        std::vector<storage_type> rows_{};
        size_type row_limit_{0};
        size_type column_limit_{0};
        std::optional<csr_storage<LayoutTag, I, V>> csr_{};
        std::vector<bool> dirty_{};
    };

    // ═════════════════════════════════════════════
    // Construction
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    matrix<L, I, V, BT, BN, LP>::matrix(size_type row_limit, size_type column_limit)
        : matrix(row_limit, column_limit, config::default_row_reserve_hint)
    {
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                     layout::ValidLayoutTag<L>
    matrix<L, I, V, BT, BN, LP>::matrix(size_type row_limit, size_type column_limit,
                                        size_type reserve_per_row)
        : mode_{config::matrix_mode::open}, rows_{}, row_limit_{row_limit},
          column_limit_{column_limit}, dirty_(row_limit, false)
    {
        rows_.reserve(row_limit_);
        for (size_type r = 0; r < row_limit_; ++r)
        {
            rows_.emplace_back(reserve_per_row, column_limit_);
        }
    }

    // ═════════════════════════════════════════════
    // Shape
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::shape() const noexcept -> shape_type
    {
        return {row_limit_, column_limit_};
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::n_rows() const noexcept -> size_type
    {
        return row_limit_;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::n_cols() const noexcept -> size_type
    {
        return column_limit_;
    }

    // ═════════════════════════════════════════════
    // Mode transitions
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::mode() const noexcept -> config::matrix_mode
    {
        return mode_;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    bool matrix<L, I, V, BT, BN, LP>::is_locked() const noexcept
    {
        return mode_ == config::matrix_mode::locked;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    bool matrix<L, I, V, BT, BN, LP>::is_open() const noexcept
    {
        return mode_ == config::matrix_mode::open;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void matrix<L, I, V, BT, BN, LP>::lock()
    {
        if (mode_ == config::matrix_mode::locked)
            return;

        // Detach stale CSR slice pointers before rebuilding.
        for (auto &r : rows_)
            r.reset_csr_slice();

        // Sort + dedup + filter each row's buffer in-place.
        for (auto &r : rows_)
            r.lock();

        mode_ = config::matrix_mode::locked;

        if constexpr (LP == config::lock_policy::compact_preserve ||
                      LP == config::lock_policy::compact_move)
        {
            // Build or merge the flat CSR (layout-appropriate).
            if (csr_)
                csr_ = merge_csr<L>(rows_, std::move(*csr_), dirty_);
            else
                csr_ = build_csr<L>(rows_);

            // Install layout-appropriate CSR slices on every row.
            const std::size_t *off = csr_->offsets.get();
            if constexpr (std::is_same_v<L, layout::tags::soa_tag>)
            {
                const I *cols_flat = csr_->cols.get();
                const V *vals_flat = csr_->vals.get();
                for (std::size_t i = 0; i < rows_.size(); ++i)
                    rows_[i].set_csr_slice(csr_slice<L, I, V>{
                        cols_flat + off[i], vals_flat + off[i], off[i + 1] - off[i]});
            }
            else // aos_tag
            {
                const auto *pairs_flat = csr_->pairs.get();
                for (std::size_t i = 0; i < rows_.size(); ++i)
                    rows_[i].set_csr_slice(csr_slice<L, I, V>{
                        pairs_flat + off[i], off[i + 1] - off[i]});
            }

            // Clear staging buffers (data now lives in the flat CSR).
            for (auto &r : rows_)
                r.clear_buffer_content();

            if constexpr (LP == config::lock_policy::compact_move)
            {
                for (auto &r : rows_)
                    r.release_buffer();
            }
        }

        // All pending changes are now committed; reset dirty flags.
        std::fill(dirty_.begin(), dirty_.end(), false);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void matrix<L, I, V, BT, BN, LP>::open()
    {
        if (mode_ == config::matrix_mode::open)
            return;
        for (auto &r : rows_)
            r.open();
        mode_ = config::matrix_mode::open;
        // csr_ is intentionally NOT reset: it holds committed history and
        // will be used as the base for merge_csr() on the next lock().
    }

    // ═════════════════════════════════════════════
    // Queries
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::row_nnz(index_type row_index) const -> size_type
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)].size();
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    bool matrix<L, I, V, BT, BN, LP>::empty() const noexcept
    {
        for (const auto &r : rows_)
        {
            if (!r.empty())
                return false;
        }
        return true;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::nnz() const noexcept -> size_type
    {
        size_type total = 0;
        for (const auto &r : rows_)
        {
            total += r.size();
        }
        return total;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::row_at(index_type row_index) const
        -> const storage_type &
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)];
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, config::lock_policy LP>
    requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> && layout::ValidLayoutTag<L>
    bool matrix<L, I, V, BT, BN, LP>::contains(index_type row_index,index_type col_index) const
    {
        validate_row_index(row_index);
        validate_col_index(col_index);
        return rows_[to_size(row_index)].contains(col_index);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::get(index_type row_index,
                                          index_type col_index) const -> value_type
    {
        validate_row_index(row_index);
        validate_col_index(col_index);
        const auto *p = rows_[to_size(row_index)].get(col_index);
        return p ? *p : traits::ValueTraits<value_type>::zero();
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::accumulate(index_type row_index) const
        -> value_type
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)].accumulate();
    }

    // ═════════════════════════════════════════════
    // Mutation (open mode)
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void matrix<L, I, V, BT, BN, LP>::insert(index_type row_index, index_type col_index,
                                             const value_type &val)
    {
        assert(mode_ == config::matrix_mode::open &&
               "matrix::insert() requires open mode");
        validate_row_index(row_index);
        validate_col_index(col_index);
        rows_[to_size(row_index)].insert(col_index, val);
        dirty_[to_size(row_index)] = true;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void matrix<L, I, V, BT, BN, LP>::clear() noexcept
    {
        assert(mode_ == config::matrix_mode::open &&
               "matrix::clear() requires open mode");
        for (auto &r : rows_)
        {
            r.clear();
        }
        std::fill(dirty_.begin(), dirty_.end(), false);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    auto matrix<L, I, V, BT, BN, LP>::row_at_mut(index_type row_index)
        -> storage_type &
    {
        assert(mode_ == config::matrix_mode::open &&
               "matrix::row_at_mut() requires open mode");
        validate_row_index(row_index);
        dirty_[to_size(row_index)] = true;
        return rows_[to_size(row_index)];
    }

    // ═════════════════════════════════════════════
    // Mode-independent
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    void matrix<L, I, V, BT, BN, LP>::swap(matrix &other) noexcept
    {
        using std::swap;
        swap(mode_, other.mode_);
        swap(rows_, other.rows_);
        swap(row_limit_, other.row_limit_);
        swap(column_limit_, other.column_limit_);
        swap(csr_, other.csr_);
        swap(dirty_, other.dirty_);
    }

    // ═════════════════════════════════════════════
    // Iteration helpers
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    template <class Func>
    void matrix<L, I, V, BT, BN, LP>::for_each_row(Func &&f) const
    {
        for (size_type i = 0; i < row_limit_; ++i)
        {
            f(rows_[i], static_cast<index_type>(i));
        }
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    template <class Func>
    void matrix<L, I, V, BT, BN, LP>::for_each_row(Func &&f)
    {
        for (size_type i = 0; i < row_limit_; ++i)
        {
            f(rows_[i], static_cast<index_type>(i));
        }
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::ValidLayoutTag<L>
    template <class Func>
    void matrix<L, I, V, BT, BN, LP>::for_each_nnz_row(Func &&f) const
    {
        for (size_type i = 0; i < row_limit_; ++i)
        {
            if (!rows_[i].empty())
            {
                f(rows_[i], static_cast<index_type>(i));
            }
        }
    }

} // namespace spira
