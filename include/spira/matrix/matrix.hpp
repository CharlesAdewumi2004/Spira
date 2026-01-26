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

        using storageType = row<LayoutTag, I, V>;

        explicit matrix(std::size_t row_limit, std::size_t column_limit);
        matrix(std::size_t row_limit, std::size_t column_limit, std::size_t reserve_per_row);

        [[nodiscard]] std::pair<std::size_t, std::size_t> get_shape() const noexcept;
        [[nodiscard]] std::size_t n_rows() const noexcept;
        [[nodiscard]] std::size_t n_cols() const noexcept;

        [[nodiscard]] std::size_t row_nnz(I row_index) const;

        [[nodiscard]] bool empty() const noexcept;

        [[nodiscard]] std::size_t nnz() const noexcept;

        [[nodiscard]] std::size_t nnz_estimate() const noexcept;

        [[nodiscard]] bool empty_estimate() const noexcept;

        [[nodiscard]] std::size_t buffer_size(I row_index) const noexcept
        {
            return rows_[static_cast<std::size_t>(row_index)].buffer_size();
        }

        [[nodiscard]] std::size_t slab_size(I row_index) const noexcept
        {
            return rows_[static_cast<std::size_t>(row_index)].slab_size();
        }

        [[nodiscard]] const storageType &getRowAt(I row_index) const;
        [[nodiscard]] storageType &getMutableRowAt(I row_index);

        void insert(I row_index, I col_index, V const &val);
        [[nodiscard]] V get(I row_index, I col_index) const;

        void clear() noexcept;

        [[nodiscard]] bool contains(I row_index, I col_index) const;

        template <class Func>
        void for_each_row(Func &&f) const;

        template <class Func>
        void for_each_nnz_row(Func &&f) const;

        void matrix_swap(matrix &other) noexcept;

        [[nodiscard]] V accumulate(I row_index) const;

        void set_mode(mode::matrix_mode new_mode);
        [[nodiscard]] mode::matrix_mode mode() const noexcept;

        void flush() const;
        void flush(I row_index) const;

        [[nodiscard]] bool is_row_dirty(I row_index) const noexcept
        {
            return rows_[static_cast<std::size_t>(row_index)].is_dirty();
        }

    private:
        void validate_row_index(I row_index) const
        {
            if (static_cast<std::size_t>(row_index) >= row_limit_)
            {
                throw std::out_of_range("matrix: row_index out of range");
            }
        }

        void validate_col_index(I col_index) const
        {
            if (static_cast<std::size_t>(col_index) >= column_limit_)
            {
                throw std::out_of_range("matrix: col_index out of range");
            }
        }

    private:
        mode::matrix_mode mode_{mode::matrix_mode::balanced};
        mutable std::vector<storageType> rows_{};
        std::size_t const row_limit_;
        std::size_t const column_limit_;
    };

    // -----------------------------
    // Construction
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    matrix<LayoutTag, I, V>::matrix(std::size_t row_limit, std::size_t column_limit)
        : matrix(row_limit, column_limit, spira::config::default_row_reserve_hint)
    {
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    matrix<LayoutTag, I, V>::matrix(std::size_t row_limit,
                                    std::size_t column_limit,
                                    std::size_t reserve_per_row)
        : mode_{mode::matrix_mode::balanced}, rows_{}, row_limit_{row_limit}, column_limit_{column_limit}
    {
        rows_.reserve(row_limit_);
        for (std::size_t r = 0; r < row_limit_; ++r)
        {
            rows_.emplace_back(reserve_per_row, column_limit_);
        }
    }

    // -----------------------------
    // Shape
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::pair<std::size_t, std::size_t> matrix<LayoutTag, I, V>::get_shape() const noexcept
    {
        return {row_limit_, column_limit_};
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t matrix<LayoutTag, I, V>::n_rows() const noexcept
    {
        return row_limit_;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t matrix<LayoutTag, I, V>::n_cols() const noexcept
    {
        return column_limit_;
    }

    // -----------------------------
    // Queries
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t matrix<LayoutTag, I, V>::row_nnz(I row_index) const
    {
        validate_row_index(row_index);
        return rows_[static_cast<std::size_t>(row_index)].size();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool matrix<LayoutTag, I, V>::empty() const noexcept
    {
        for (auto const &r : rows_)
        {
            if (r.size() != 0)
            {
                return false;
            }
        }
        return true;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t matrix<LayoutTag, I, V>::nnz() const noexcept
    {
        std::size_t entries = 0;
        for (auto const &r : rows_)
        {
            entries += r.size();
        }
        return entries;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t matrix<LayoutTag, I, V>::nnz_estimate() const noexcept
    {
        std::size_t entries = 0;
        for (auto const &r : rows_)
        {
            entries += r.slab_size();
            entries += r.buffer_size();
        }
        return entries;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool matrix<LayoutTag, I, V>::empty_estimate() const noexcept
    {
        for (auto const &r : rows_)
        {
            if (r.slab_size() != 0 || r.buffer_size() != 0)
            {
                return false;
            }
        }
        return true;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    [[nodiscard]] const row<LayoutTag, I, V> &matrix<LayoutTag, I, V>::getRowAt(I row_index) const
    {
        validate_row_index(row_index);
        return rows_[row_index];
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    [[nodiscard]] row<LayoutTag, I, V> &matrix<LayoutTag, I, V>::getMutableRowAt(I row_index)
    {
        validate_row_index(row_index);
        return rows_[row_index];
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::insert(I row_index, I col_index, V const &val)
    {
        validate_row_index(row_index);
        validate_col_index(col_index);

        rows_[static_cast<std::size_t>(row_index)].insert(col_index, val);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    V matrix<LayoutTag, I, V>::get(I row_index, I col_index) const
    {
        validate_row_index(row_index);
        validate_col_index(col_index);

        V const *p = rows_[static_cast<std::size_t>(row_index)].get(col_index);
        return p ? *p : spira::traits::ValueTraits<V>::zero();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool matrix<LayoutTag, I, V>::contains(I row_index, I col_index) const
    {
        validate_row_index(row_index);
        validate_col_index(col_index);

        return rows_[static_cast<std::size_t>(row_index)].contains(col_index);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Func>
    void matrix<LayoutTag, I, V>::for_each_row(Func &&f) const
    {
        for (std::size_t i = 0; i < row_limit_; ++i)
        {
            f(rows_[i], i);
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Func>
    void matrix<LayoutTag, I, V>::for_each_nnz_row(Func &&f) const
    {
        for (std::size_t i = 0; i < row_limit_; ++i)
        {
            if (!rows_[i].empty())
            {
                f(rows_[i], i);
            }
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::matrix_swap(matrix &other) noexcept
    {
        using std::swap;
        swap(rows_, other.rows_);
        swap(mode_, other.mode_);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    V matrix<LayoutTag, I, V>::accumulate(I row_index) const
    {
        validate_row_index(row_index);
        return rows_[static_cast<std::size_t>(row_index)].accumulate();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    mode::matrix_mode matrix<LayoutTag, I, V>::mode() const noexcept
    {
        return mode_;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void matrix<LayoutTag, I, V>::set_mode(mode::matrix_mode new_mode)
    {
        mode_ = new_mode;
        for (auto &r : rows_)
        {
            r.set_mode(new_mode);
        }
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
    void matrix<LayoutTag, I, V>::flush(I row_index) const
    {
        validate_row_index(row_index);

        auto &r = rows_[static_cast<std::size_t>(row_index)];
        if (r.is_dirty())
        {
            r.flush();
        }
    }

}
