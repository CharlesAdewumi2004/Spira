#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spira/config.hpp>
#include <spira/matrix/mode/matrix_mode.hpp>
#include <spira/matrix/row.hpp>
#include <spira/traits.hpp>

namespace spira
{

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    class matrix
    {
    public:
        using layout_policy = layout::of::storage_of_t<LayoutTag, I, V>;
        using index_type = I;
        using value_type = V;

        using storage_type = row<LayoutTag, I, V>;
        using size_type = std::size_t;
        using shape_type = std::pair<size_type, size_type>;

        // -----------------------------
        // Construction / lifetime
        // -----------------------------

        explicit matrix(size_type row_limit, size_type column_limit);
        matrix(size_type row_limit, size_type column_limit, size_type reserve_per_row);

        ~matrix() = default;
        matrix(const matrix &) = default;
        matrix(matrix &&) noexcept = default;
        matrix &operator=(const matrix &) = default;
        matrix &operator=(matrix &&) noexcept = default;

        // -----------------------------
        // Shape
        // -----------------------------

        [[nodiscard]] shape_type shape() const noexcept;
        [[nodiscard]] size_type n_rows() const noexcept;
        [[nodiscard]] size_type n_cols() const noexcept;

        // -----------------------------
        // Queries
        // -----------------------------

        [[nodiscard]] size_type row_nnz(index_type row_index) const;
        [[nodiscard]] bool empty() const noexcept;

        [[nodiscard]] size_type nnz() const noexcept;
        [[nodiscard]] size_type nnz_estimate() const noexcept;
        [[nodiscard]] bool empty_estimate() const noexcept;

        [[nodiscard]] size_type buffer_size(index_type row_index) const noexcept;
        [[nodiscard]] size_type slab_size(index_type row_index) const noexcept;

        [[nodiscard]] const storage_type &row_at(index_type row_index) const;
        [[nodiscard]] storage_type &row_at_mut(index_type row_index);

        [[nodiscard]] bool contains(index_type row_index, index_type col_index) const;
        [[nodiscard]] value_type get(index_type row_index, index_type col_index) const;

        [[nodiscard]] value_type accumulate(index_type row_index) const;

        [[nodiscard]] mode::matrix_mode mode() const noexcept;
        void set_mode(mode::matrix_mode new_mode);

        [[nodiscard]] bool is_row_dirty(index_type row_index) const noexcept;

        // -----------------------------
        // Mutation
        // -----------------------------

        void insert(index_type row_index, index_type col_index, const value_type &val);
        void clear() noexcept;

        void flush() const;
        void flush(index_type row_index) const;

        void swap(matrix &other) noexcept;

        // -----------------------------
        // Iteration helpers
        // -----------------------------

        template <class Func>
        void for_each_row(Func &&f) const;

        template <class Func>
        void for_each_nnz_row(Func &&f) const;

        // -----------------------------
        // Arithmetic 
        // -----------------------------

        // add/sub
        matrix operator+(const matrix &other) const;
        matrix operator-(const matrix &other) const;
        matrix &operator+=(const matrix &other);
        matrix &operator-=(const matrix &other);

        // spgemm
        matrix operator*(const matrix &other) const;
        matrix &operator*=(const matrix &other);

        // spmv
        std::vector<value_type> operator*(const std::vector<value_type> &x) const;

        // scalar
        matrix operator*(value_type s) const;
        matrix &operator*=(value_type s);
        matrix operator/(value_type s) const;
        matrix &operator/=(value_type s);

        // transpose
        matrix operator~() const;

    private:
        static constexpr size_type to_size(index_type i) noexcept
        {
            return static_cast<size_type>(i);
        }

        void validate_row_index(index_type row_index) const
        {
            if (to_size(row_index) >= row_limit_)
            {
                throw std::out_of_range("spira::matrix: row_index out of range");
            }
        }

        void validate_col_index(index_type col_index) const
        {
            if (to_size(col_index) >= column_limit_)
            {
                throw std::out_of_range("spira::matrix: col_index out of range");
            }
        }

    private:
        mode::matrix_mode mode_{mode::matrix_mode::balanced};
        mutable std::vector<storage_type> rows_{};
        size_type row_limit_{0};
        size_type column_limit_{0};
    };

    // -----------------------------
    // Construction
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    matrix<LayoutTag, I, V>::matrix(size_type row_limit, size_type column_limit)
        : matrix(row_limit, column_limit, spira::config::default_row_reserve_hint) {}

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    matrix<LayoutTag, I, V>::matrix(size_type row_limit, size_type column_limit, size_type reserve_per_row)
        : mode_{mode::matrix_mode::balanced}, rows_{}, row_limit_{row_limit}, column_limit_{column_limit}
    {
        rows_.reserve(row_limit_);
        for (size_type r = 0; r < row_limit_; ++r)
        {
            rows_.emplace_back(reserve_per_row, column_limit_);
        }
    }

    // -----------------------------
    // Shape
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::shape() const noexcept -> shape_type
    {
        return {row_limit_, column_limit_};
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::n_rows() const noexcept -> size_type
    {
        return row_limit_;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::n_cols() const noexcept -> size_type
    {
        return column_limit_;
    }

    // -----------------------------
    // Queries
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::row_nnz(index_type row_index) const -> size_type
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)].size();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool matrix<LayoutTag, I, V>::empty() const noexcept
    {
        for (const auto &r : rows_)
        {
            if (!r.empty())
            {
                return false;
            }
        }
        return true;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::nnz() const noexcept -> size_type
    {
        size_type entries = 0;
        for (const auto &r : rows_)
        {
            entries += r.size();
        }
        return entries;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::nnz_estimate() const noexcept -> size_type
    {
        size_type entries = 0;
        for (const auto &r : rows_)
        {
            entries += r.slab_size();
            entries += r.buffer_size();
        }
        return entries;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool matrix<LayoutTag, I, V>::empty_estimate() const noexcept
    {
        for (const auto &r : rows_)
        {
            if (r.slab_size() != 0 || r.buffer_size() != 0)
            {
                return false;
            }
        }
        return true;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::buffer_size(index_type row_index) const noexcept -> size_type
    {
        return rows_[to_size(row_index)].buffer_size();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::slab_size(index_type row_index) const noexcept -> size_type
    {
        return rows_[to_size(row_index)].slab_size();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::row_at(index_type row_index) const -> const storage_type &
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)];
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::row_at_mut(index_type row_index) -> storage_type &
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)];
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::insert(index_type row_index, index_type col_index, const value_type &val)
    {
        validate_row_index(row_index);
        validate_col_index(col_index);
        rows_[to_size(row_index)].insert(col_index, val);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::get(index_type row_index, index_type col_index) const -> value_type
    {
        validate_row_index(row_index);
        validate_col_index(col_index);

        const value_type *p = rows_[to_size(row_index)].get(col_index);
        return p ? *p : spira::traits::ValueTraits<value_type>::zero();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool matrix<LayoutTag, I, V>::contains(index_type row_index, index_type col_index) const
    {
        validate_row_index(row_index);
        validate_col_index(col_index);
        return rows_[to_size(row_index)].contains(col_index);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::accumulate(index_type row_index) const -> value_type
    {
        validate_row_index(row_index);
        return rows_[to_size(row_index)].accumulate();
    }

    // -----------------------------
    // Mode / dirty / flush
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    auto matrix<LayoutTag, I, V>::mode() const noexcept -> mode::matrix_mode
    {
        return mode_;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::set_mode(mode::matrix_mode new_mode)
    {
        if(new_mode == mode_){
            return;
        }
        mode_ = new_mode;
        for (auto &r : rows_)
        {
            r.set_mode(new_mode);
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool matrix<LayoutTag, I, V>::is_row_dirty(index_type row_index) const noexcept
    {
        return rows_[to_size(row_index)].is_dirty();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::clear() noexcept
    {
        for (auto &r : rows_)
        {
            r.clear();
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::flush() const
    {
        for (auto &r : rows_)
        {
            if (r.is_dirty())
            {
                r.flush();
            }
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::flush(index_type row_index) const
    {
        validate_row_index(row_index);

        auto &r = rows_[to_size(row_index)];
        if (r.is_dirty())
        {
            r.flush();
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::swap(matrix &other) noexcept
    {
        using std::swap;
        swap(rows_, other.rows_);
        swap(mode_, other.mode_);
        swap(row_limit_, other.row_limit_);
        swap(column_limit_, other.column_limit_);
    }

    // -----------------------------
    // Iteration helpers
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Func>
    void matrix<LayoutTag, I, V>::for_each_row(Func &&f) const
    {
        for (size_type i = 0; i < row_limit_; ++i)
        {
            f(rows_[i], i);
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Func>
    void matrix<LayoutTag, I, V>::for_each_nnz_row(Func &&f) const
    {
        for (size_type i = 0; i < row_limit_; ++i)
        {
            if (!rows_[i].empty())
            {
                f(rows_[i], i);
            }
        }
    }

}
