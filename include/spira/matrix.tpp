#pragma once
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace spira {

template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize>
matrix<V, I, rowSize, colSize>::matrix() noexcept {
    _matrix.resize(rowSize);
}

template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize>
bool matrix<V, I, rowSize, colSize>::setRow(I rowIndex, std::vector<element<V, I>> elems) {
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

template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize>
void matrix<V, I, rowSize, colSize>::printRow(I rowIndex) const noexcept {
    if (rowIndex >= rowSize) {
        std::cout << "(row out of range)\n";
        return;
    }
    if (_matrix[rowIndex].empty()) {
        std::cout << "(empty matrix)\n";
    }
    for (const auto& e : _matrix[rowIndex]) {
        std::cout << "(col=" << e.col << ", val=" << e.value << ") ";
    }
    std::cout << "\n";
}

template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize>
std::optional<V> matrix<V, I, rowSize, colSize>::at(I rowIndex, I colIndex) const {
    if (rowIndex >= rowSize) {
        throw std::out_of_range("Row index out of range");
    }
    if (colIndex >= colSize) {
        throw std::out_of_range("Column index out of range");
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

template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize>
void matrix<V, I, rowSize, colSize>::printMatrix() const noexcept {
    for (I i = 0; i < rowSize; ++i) {
        if (_matrix[i].empty()) {
            for (I j = 0; j < colSize; ++j) {
                std::cout << " 0 ";
            }
            std::cout << '\n';
        }else {
            for (I j = 0; j < colSize; ++j) {
                if (auto result = at(i, j)) {
                    std::cout << " " <<  *result << " ";
                }else {
                    std::cout << " " <<  0 << " ";
                }
            }
            std::cout << '\n';
        }
    }
}

}
