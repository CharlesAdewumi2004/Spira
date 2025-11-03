#pragma once
#include "row.hpp"


namespace spira {
template<class Layout, concepts::Indexable I, concepts::Valueable V>
class matrix {

    public:
        using layout_policy = layout::of::storage_of_t<Layout, I, V>;
        using index_type = I;
        using value_type = V;

        explicit matrix(size_t const row_limit, size_t const column_limit) : _row_limit(row_limit), _column_limit(column_limit) {
            _rows.reserve(_row_limit);
            for (size_t i = 0; i < _row_limit; i++) {
                _rows.emplace_back(column_limit,column_limit);
            }

        }
        [[nodiscard]] std::pair<size_t, size_t> get_shape() const noexcept {
            return std::make_pair(_row_limit, _column_limit);
        }
        [[nodiscard]] size_t n_rows() const noexcept {
            return _row_limit;
        }
        [[nodiscard]] size_t n_cols() const noexcept {
            return _column_limit;
        }
        [[nodiscard]] size_t row_nnz(I row_index) const {
            if (row_index >= _row_limit) {
                throw std::out_of_range("Input is out of range of the matrix");
            }
            return _rows[row_index].size();
        }
        [[nodiscard]] bool empty() const noexcept{
            for (auto const &row : _rows) {
                if (row.size() != 0) {
                    return false;
                }
            }
            return true;
        }
        [[nodiscard]] size_t nnz() const noexcept {
            size_t entries = 0;
            for (auto const &row : _rows) {
                entries += row.size();
            }
            return entries;
        }
        void add(I row_index, I col_index, V &&val) {
            if (row_index >= _row_limit) {
                throw std::out_of_range("Row index out of range");
            }
            if (col_index >= _column_limit) {
                throw std::out_of_range("Column index out of range");
            }
            _rows[row_index].add(col_index, val);
        }

        [[nodiscard]] V get(I row_index, I col_index) const {
            if (row_index >= _row_limit) {
                throw std::out_of_range("Row index out of range");
            }
            if (col_index >= _column_limit) {
                throw std::out_of_range("Column index out of range");
            }
            V const *val = _rows[row_index].get(col_index);
            if (val == nullptr) {
                return spira::traits::ValueTraits<V>::zero();
            }
            return *val;
        }

        void clear() noexcept {
            for (auto &row : _rows) {
                row.clear();
            }
        }

        template<class PairRange>
        void set_row(I row_index, const PairRange& elems) {
            if (row_index >= _row_limit) {
                throw std::out_of_range("Row index out of range");
            }
            _rows[row_index].set_row(elems);
        }

        [[nodiscard]] bool contains(I row_index, I col_index) const {
            if (row_index >= _row_limit) {
                throw std::out_of_range("Row index out of range");
            }
            if (col_index >= _column_limit) {
                throw std::out_of_range("Column index out of range");
            }
            return _rows[row_index].contains(col_index);
        }

        void remove(I row_index, I col_index) {
            if (row_index >= _row_limit) {
                throw std::out_of_range("Row index out of range");
            }
            if (col_index >= _column_limit) {
                throw std::out_of_range("Column index out of range");
            }
            _rows[row_index].remove(col_index);
        }

    private:
        std::vector<row<Layout, I, V>> _rows;
        size_t const _row_limit;
        size_t const _column_limit;

};



}

