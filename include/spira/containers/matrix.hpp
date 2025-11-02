#pragma once
#include "row.hpp"


namespace spira {
template<class Layout, concepts::Indexable I, concepts::Valueable V>
class matrix {
    public:
        explicit matrix(size_t const row_limit, size_t const column_limit) : _row_limit(row_limit), _column_limit(column_limit) {
            _rows.resize(_row_limit);
        }
    private:
        std::vector<row<Layout, I, V>> _rows;
        size_t const _row_limit;
        size_t const _column_limit;

};



}

