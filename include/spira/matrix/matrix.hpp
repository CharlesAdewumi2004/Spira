#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/matrix/layouts/layout_base.hpp>
#include <spira/matrix/layouts/layout_of.hpp>
#include <spira/matrix/row.hpp>
#include <spira/traits.hpp>

namespace spira
{

    template <class LayoutTag, concepts::Indexable I = uint32_t,
              concepts::Valueable V = double,
              class BufferTag = buffer::tags::array_buffer<LayoutTag>,
              std::size_t BufferN = 64>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I,V> &&
                 layout::Layout<layout::detail::storage_of_t<LayoutTag, I, V>, I, V>
    class matrix
    {
    public:
        using layout_policy = layout::detail::storage_of_t<LayoutTag, I, V>;
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
    };

    // ═════════════════════════════════════════════
    // Construction
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    matrix<L, I, V, BT, BN>::matrix(size_type row_limit, size_type column_limit)
        : matrix(row_limit, column_limit, config::default_row_reserve_hint)
    {
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                     layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    matrix<L, I, V, BT, BN>::matrix(size_type row_limit, size_type column_limit,
                                    size_type reserve_per_row)
        : mode_{config::matrix_mode::open}, rows_{}, row_limit_{row_limit},
          column_limit_{column_limit}
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
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::shape() const noexcept -> shape_type
    {
        return {row_limit_, column_limit_};
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::n_rows() const noexcept -> size_type
    {
        return row_limit_;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::n_cols() const noexcept -> size_type
    {
        return column_limit_;
    }

    // ═════════════════════════════════════════════
    // Mode transitions
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::mode() const noexcept -> config::matrix_mode
    {
        return mode_;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    bool matrix<L, I, V, BT, BN>::is_locked() const noexcept
    {
        return mode_ == config::matrix_mode::locked;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    bool matrix<L, I, V, BT, BN>::is_open() const noexcept
    {
        return mode_ == config::matrix_mode::open;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    void matrix<L, I, V, BT, BN>::lock()
    {
        if (mode_ == config::matrix_mode::locked)
            return;
        for (auto &r : rows_)
        {
            r.lock();
        }
        mode_ = config::matrix_mode::locked;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    void matrix<L, I, V, BT, BN>::open()
    {
        if (mode_ == config::matrix_mode::open)
            return;
        for (auto &r : rows_)
        {
            r.open();
        }
        mode_ = config::matrix_mode::open;
    }

    // ═════════════════════════════════════════════
    // Queries
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::row_nnz(index_type row_index) const -> size_type
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)].size();
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    bool matrix<L, I, V, BT, BN>::empty() const noexcept
    {
        for (const auto &r : rows_)
        {
            if (!r.empty())
                return false;
        }
        return true;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::nnz() const noexcept -> size_type
    {
        size_type total = 0;
        for (const auto &r : rows_)
        {
            total += r.size();
        }
        return total;
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::row_at(index_type row_index) const
        -> const storage_type &
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)];
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    bool matrix<L, I, V, BT, BN>::contains(index_type row_index,
                                           index_type col_index) const
    {
        validate_row_index(row_index);
        validate_col_index(col_index);
        return rows_[to_size(row_index)].contains(col_index);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::get(index_type row_index,
                                      index_type col_index) const -> value_type
    {
        validate_row_index(row_index);
        validate_col_index(col_index);
        const auto *p = rows_[to_size(row_index)].get(col_index);
        return p ? *p : traits::ValueTraits<value_type>::zero();
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::accumulate(index_type row_index) const
        -> value_type
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)].accumulate();
    }

    // ═════════════════════════════════════════════
    // Mutation (open mode)
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    void matrix<L, I, V, BT, BN>::insert(index_type row_index, index_type col_index,
                                         const value_type &val)
    {
        assert(mode_ == config::matrix_mode::open &&
               "matrix::insert() requires open mode");
        validate_row_index(row_index);
        validate_col_index(col_index);
        rows_[to_size(row_index)].insert(col_index, val);
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    void matrix<L, I, V, BT, BN>::clear() noexcept
    {
        assert(mode_ == config::matrix_mode::open &&
               "matrix::clear() requires open mode");
        for (auto &r : rows_)
        {
            r.clear();
        }
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    auto matrix<L, I, V, BT, BN>::row_at_mut(index_type row_index)
        -> storage_type &
    {
        assert(mode_ == config::matrix_mode::open &&
               "matrix::row_at_mut() requires open mode");
        validate_row_index(row_index);
        return rows_[to_size(row_index)];
    }

    // ═════════════════════════════════════════════
    // Mode-independent
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    void matrix<L, I, V, BT, BN>::swap(matrix &other) noexcept
    {
        using std::swap;
        swap(mode_, other.mode_);
        swap(rows_, other.rows_);
        swap(row_limit_, other.row_limit_);
        swap(column_limit_, other.column_limit_);
    }

    // ═════════════════════════════════════════════
    // Iteration helpers
    // ═════════════════════════════════════════════

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    template <class Func>
    void matrix<L, I, V, BT, BN>::for_each_row(Func &&f) const
    {
        for (size_type i = 0; i < row_limit_; ++i)
        {
            f(rows_[i], static_cast<index_type>(i));
        }
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    template <class Func>
    void matrix<L, I, V, BT, BN>::for_each_row(Func &&f)
    {
        for (size_type i = 0; i < row_limit_; ++i)
        {
            f(rows_[i], static_cast<index_type>(i));
        }
    }

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN>
        requires buffer::Buffer<buffer::traits::traits_of_type<BT, I, V, BN>, I, V> &&
                 layout::Layout<layout::detail::storage_of_t<L, I, V>, I, V>
    template <class Func>
    void matrix<L, I, V, BT, BN>::for_each_nnz_row(Func &&f) const
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
