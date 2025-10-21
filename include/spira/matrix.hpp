#pragma once
#include <vector>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "element.hpp"

namespace spira {
    template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize>
    class matrix {
    private:
        std::vector<std::vector<element<V, I>>> _matrix;

    public:
        matrix() {
            _matrix.resize(rowSize);
        }

        bool setRow(I rowIndex, std::vector<element<V, I>>&& elems) {
            if (rowIndex >= rowSize) {
                throw std::out_of_range("Row index out of range");
            }
            if (!_matrix[rowIndex].empty()) {
                return false;
            }
            if (elems.size() > colSize) {
                throw std::out_of_range("Input row too large");
            }
            for (const auto& e : elems) {
                if (e.col >= colSize) {
                    throw std::out_of_range("Element column out of range");
                }
            }
            std::sort(elems.begin(), elems.end(),
                      [](const element<V, I>& a, const element<V, I>& b) {
                          return a.col < b.col;
                      });
            auto dup = std::adjacent_find(elems.begin(), elems.end(),
                                          [](const auto& a, const auto& b) {
                                              return a.col == b.col;
                                          });
            if (dup != elems.end()) {
                throw std::invalid_argument("Duplicate column indices in row");
            }

            _matrix[rowIndex] = std::move(elems);
            return true;
        }

        void printRow(I rowIndex) const {
            if (rowIndex >= rowSize) {
                std::cout << "(row out of range)\n";
                return;
            }
            for (const auto& e : _matrix[rowIndex]) {
                std::cout << "(col=" << e.col << ", val=" << e.value << ") ";
            }
            std::cout << "\n";
        }

        std::optional<V> at(I rowIndex, I colIndex) const {
            if (rowIndex >= rowSize) {
                return std::nullopt;
            }
            if (colIndex >= colSize) {
                return std::nullopt;
            }
            const auto& row = _matrix[rowIndex];
            auto it = std::lower_bound(row.begin(), row.end(), colIndex,
                                       [](const element<V, I>& e, I c) {
                                           return e.col < c;
                                       });
            if (it != row.end() && it->col == colIndex) {
                return it->value;
            }
            return std::nullopt;
        }
    };
}

