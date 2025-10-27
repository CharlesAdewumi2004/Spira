#pragma once

#include <vector>
#include <algorithm>

#include "spira/concepts.hpp"

namespace spira::layout {
    template<concepts::Indexable I, concepts::Valueable V>
    class soa {
        private:
            std::vector<I> columns;
            std::vector<V> values;

        public:
            [[nodiscard]] bool empty() const noexcept {
                return  columns.empty();
            }
            [[nodiscard]] size_t size() const noexcept {
                return columns.size();
            }
            void clear() noexcept {
                columns.clear();
                values.clear();
            }
            void reserve(size_t size) {
                columns.reserve(size);
                values.reserve(size);
            }
            [[nodiscard]] size_t capacity() const noexcept {
                return columns.capacity();
            }
            [[nodiscard]] I key_at(size_t index) const noexcept {
                return columns[index];
            }
            [[nodiscard]] V& value_at(size_t pos) noexcept {
                    return values[pos];
                }

            [[nodiscard]] const V& value_at(size_t pos) const noexcept {
                    return values[pos];
                }
            void set_at(size_t index, I col,  V val)  {
                columns[index] = col;
                values[index] = val;
            }
            void insert_at(size_t index, I col, V val)  {
                columns.insert(columns.begin() + index, col);
                values.insert(values.begin() + index, val);
            }
            void erase_at(size_t index)  {
                columns.erase(columns.begin() + index);
                values.erase(values.begin() + index);
            }
            [[nodiscard]] size_t lower_bound(I col) const noexcept {
                auto it = std::lower_bound(columns.begin(), columns.end(), col);
                if (it == columns.end()) {
                    return columns.size();
                }else {
                    return static_cast<size_t>(std::distance(columns.begin(), it));
                }
            }
    };
}