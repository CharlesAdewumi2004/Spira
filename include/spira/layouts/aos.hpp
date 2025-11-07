#pragma once
#include <vector>
#include <algorithm>

#include "spira/concepts.hpp"

namespace spira::layout {
    template<concepts::Indexable I, concepts::Valueable V>
    struct elementPair {
        I column;
        V value;
    };

    template<concepts::Indexable I, concepts::Valueable V>
    class aos {
        public:
            struct Iterator {
                using value_type        = elementPair<I,V>;
                using difference_type   = std::ptrdiff_t;
                using pointer           = value_type*;
                using reference         = value_type&;
                using iterator_category = std::contiguous_iterator_tag;

                pointer ptr{};

                explicit constexpr Iterator(pointer p) noexcept : ptr(p) {}

                constexpr reference operator*() const noexcept { return *ptr; }
                constexpr pointer   operator->() const noexcept { return  ptr; }
                constexpr reference operator[](difference_type n) const noexcept { return ptr[n]; }

                constexpr Iterator& operator++() noexcept { ++ptr; return *this; }
                constexpr Iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
                constexpr Iterator& operator--() noexcept { --ptr; return *this; }
                constexpr Iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }

                constexpr Iterator& operator+=(difference_type n) noexcept { ptr += n; return *this; }
                constexpr Iterator& operator-=(difference_type n) noexcept { ptr -= n; return *this; }
                friend constexpr Iterator operator+(Iterator it, difference_type n) noexcept { it += n; return it; }
                friend constexpr Iterator operator+(difference_type n, Iterator it) noexcept { it += n; return it; }
                friend constexpr Iterator operator-(Iterator it, difference_type n) noexcept { it -= n; return it; }

                friend constexpr difference_type operator-(Iterator a, Iterator b) noexcept {
                    return a.ptr - b.ptr;
                }

                friend constexpr bool operator==(const Iterator&, const Iterator&) noexcept = default;
                friend constexpr auto operator<=>(const Iterator&, const Iterator&) noexcept = default;
            };

            [[nodiscard]] bool empty() const noexcept {
                return elements.empty();
            }
            void reserve(size_t size) {
                elements.reserve(size);
            }
            [[nodiscard]] size_t capacity() const noexcept {
                return elements.capacity();
            }
            [[nodiscard]] size_t size() const noexcept {
                return elements.size();
            }
            void clear() {
                elements.clear();
            }
            I key_at(size_t index) const noexcept {
                return elements[index].column;
            }
            [[nodiscard]] V& value_at(size_t pos) noexcept {
                    return elements[pos].value;
                }

            [[nodiscard]] const V& value_at(size_t pos) const noexcept {
                    return elements[pos].value;
                }
            void set_at(size_t index, I col, const V& val) {
                elements.insert(elements.begin() + index, elementPair<I, V>{col, val});
            }
            void insert_at(size_t index, I col, const V& val) {
                elements.insert(elements.begin() + index, elementPair<I,V>{col, val});
            }
            void erase_at(size_t index) {
                elements.erase(elements.begin() + index);
            }
            [[nodiscard]] size_t lower_bound(I col) const noexcept {
                auto it = std::lower_bound(
                    elements.begin(), elements.end(), col,
                    [](auto const& e, I key){ return e.column < key; }
                );
                return static_cast<size_t>(distance(elements.begin(), it));
            }

            Iterator begin() {
                return Iterator(elements.data());
            }

            Iterator end() {
                return Iterator(elements.data() + elements.size());
            }

    private:
        std::vector<elementPair<I, V>> elements;
    };
}
