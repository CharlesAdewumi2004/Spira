#pragma once

#include <cstddef>
#include <type_traits>

namespace spira::buffer
{

    template <class Derived, class I, class V>
    class base_buffer
    {
    public:
        [[nodiscard]] bool empty() const noexcept { return self().empty_impl(); }
        [[nodiscard]] size_t size() const noexcept { return self().size_impl(); }
        [[nodiscard]] size_t remaining_capacity() const noexcept
        {
            return self().remaining_capacity_impl();
        }

        void clear() noexcept { self().clear_impl(); }

        void push_back(const I &col,
                       const V &val) noexcept(std::is_nothrow_copy_assignable_v<I> &&
                                              std::is_nothrow_copy_assignable_v<V>)
        {
            self().push_back_impl(col, val);
        }

        [[nodiscard]] bool contains(I col) const noexcept
        {
            return self().contains_impl(col);
        }

        [[nodiscard]] const V *get_ptr(I col) const noexcept
        {
            return self().get_ptr_impl(col);
        }

        [[nodiscard]] V accumulate() const
        {
            return self().accumulate_impl();
        }

        /// Sort by column, deduplicate (last-write wins), and filter zero values.
        /// After this call the buffer is sorted, unique, and zero-free.
        void sort_and_dedup() { self().sort_and_dedup(); }

        [[nodiscard]] auto begin() noexcept { return self().begin_impl(); }
        [[nodiscard]] auto end() noexcept { return self().end_impl(); }
        [[nodiscard]] auto begin() const noexcept { return self().begin_impl(); }
        [[nodiscard]] auto end() const noexcept { return self().end_impl(); }
        [[nodiscard]] auto cbegin() const noexcept { return self().begin_impl(); }
        [[nodiscard]] auto cend() const noexcept { return self().end_impl(); }

    private:
        Derived &self() { return static_cast<Derived &>(*this); }
        const Derived &self() const { return static_cast<const Derived &>(*this); }
    };

    template <class T, class I, class V>
    concept Buffer = std::is_base_of_v<base_buffer<T, I, V>, T>;

} // namespace spira::buffer