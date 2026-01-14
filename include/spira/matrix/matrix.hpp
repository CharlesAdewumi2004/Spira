#pragma once

#include <spira/matrix/row.hpp>
#include <spira/matrix/mode/matrix_mode.hpp>

namespace spira
{

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    class matrix
    {
    public:
        using layout_policy = layout::of::storage_of_t<Layout, I, V>;
        using index_type = I;
        using value_type = V;

        explicit matrix(size_t const row_limit, size_t const column_limit);

        std::pair<size_t, size_t> get_shape() const noexcept;
        size_t n_rows() const noexcept;
        size_t n_cols() const noexcept;
        size_t row_nnz(I row_index) const;
        bool empty() const noexcept;
        size_t nnz() const noexcept;

        void add(I row_index, I col_index, const V &val);

        V get(I row_index, I col_index) const;

        void clear() noexcept;

        template <class PairRange>
        void set_row(I row_index, const PairRange &elems);

        bool contains(I row_index, I col_index) const;

        void remove(I row_index, I col_index);

        template <class Func>
        void for_each_row(Func &&f) const;

        V accumlate(I row_index) const;

        void set_mode(mode::matrix_mode new_mode);
        mode::matrix_mode mode() const noexcept;

    private:
        mode::matrix_mode mode_ = mode::matrix_mode::balanced;
        std::vector<row<Layout, I, V>> row_;
        size_t const row_limit_;
        size_t const column_limit_;
    };

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    matrix<Layout, I, V>::matrix(size_t const row_limit, size_t const column_limit)
        : row_limit_(row_limit),
          column_limit_(column_limit)
    {
        mode_ = mode::matrix_mode::balanced;
        row_.reserve(row_limit_);
        for (size_t i = 0; i < row_limit_; i++)
        {
            row_.emplace_back(column_limit, column_limit);
        }
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    std::pair<size_t, size_t> matrix<Layout, I, V>::get_shape() const noexcept
    {
        return std::make_pair(row_limit_, column_limit_);
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    size_t matrix<Layout, I, V>::n_rows() const noexcept
    {
        return row_limit_;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    size_t matrix<Layout, I, V>::n_cols() const noexcept
    {
        return column_limit_;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    size_t matrix<Layout, I, V>::row_nnz(I row_index) const
    {
        if (row_index >= row_limit_)
        {
            throw std::out_of_range("Input is out of range of the matrix");
        }
        return row_[row_index].size();
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    bool matrix<Layout, I, V>::empty() const noexcept
    {
        for (auto const &row : row_)
        {
            if (row.size() != 0)
            {
                return false;
            }
        }
        return true;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    size_t matrix<Layout, I, V>::nnz() const noexcept
    {
        size_t entries = 0;
        for (auto const &row : row_)
        {
            entries += row.size();
        }
        return entries;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void matrix<Layout, I, V>::add(I row_index, I col_index, const V &val)
    {
        if (row_index >= row_limit_)
        {
            throw std::out_of_range("Row index out of range");
        }
        if (col_index >= column_limit_)
        {
            throw std::out_of_range("Column index out of range");
        }
        row_[row_index].add(col_index, val);
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    V matrix<Layout, I, V>::get(I row_index, I col_index) const
    {
        if (row_index >= row_limit_)
        {
            throw std::out_of_range("Row index out of range");
        }
        if (col_index >= column_limit_)
        {
            throw std::out_of_range("Column index out of range");
        }

        V const *val = row_[row_index].get(col_index);
        if (val == nullptr)
        {
            return spira::traits::ValueTraits<V>::zero();
        }
        return *val;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void matrix<Layout, I, V>::clear() noexcept
    {
        for (auto &row : row_)
        {
            row.clear();
        }
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    bool matrix<Layout, I, V>::contains(I row_index, I col_index) const
    {
        if (row_index >= row_limit_)
        {
            throw std::out_of_range("Row index out of range");
        }
        if (col_index >= column_limit_)
        {
            throw std::out_of_range("Column index out of range");
        }
        return row_[row_index].contains(col_index);
    }


    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    template <class Func>
    void matrix<Layout, I, V>::for_each_row(Func &&f) const
    {
        for (size_t i = 0; i < row_limit_; i++)
        {
            f(row_[i], i);
        }
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    V matrix<Layout, I, V>::accumlate(I row_index) const{
        if(row_index >= row_limit_){
            throw std::out_of_range("Row index out of range");
        }
        return row_[row_index].accumlate();
    }
    
    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    mode::matrix_mode matrix<Layout, I, V>::mode() const noexcept{
            return mode_;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void matrix<Layout, I, V>::set_mode(mode::matrix_mode new_mode){
        mode_ = new_mode;
        for(auto &row : row_){
            row.set_mode(new_mode);
        }
    }

}
