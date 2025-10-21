#pragma once

#include <vector>

#include "row.hpp"

namespace spira {
    template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize >
    class matrix {
        private:
            std::vector<row<V, I, colSize>> _rows;
        public:
            matrix() noexcept{
                _rows.resize(rowSize);
            }

            bool tryFillRow(I atRow ,std::vector<element::element<V,I>>&& elems) {
                rangeCheckRow(atRow);
                if (!_rows[atRow].setRow(std::move(elems))) {
                    return false;
                }
                return true;
            }

            void printRow(I atRow) const noexcept {
                _rows[atRow].printRow();
            }

            void rangeCheckRow(I rowIndex) {
                if (rowIndex >= rowSize) {
                    throw std::range_error("row index is out of range");
                }
            }

            void rangeCheckCol(I colIndex) {
                if (colIndex >= colSize) {
                    throw std::range_error("column index is out of range");
                }
            }
    };
}