#pragma once

#include <cstddef>
#include <memory>
#include <new>

#include <spira/matrix/layout/layout_tags.hpp>

namespace spira
{

    // Primary template declarations — specialisations live in csr_storage_{soa,aos}.hpp
    template <class LayoutTag, class I, class V>
    struct csr_storage;

    template <class LayoutTag, class I, class V>
    struct csr_slice;

    inline constexpr std::size_t csr_alignment = 64;

    namespace detail
    {

        struct csr_aligned_deleter
        {
            void operator()(void *p) const noexcept
            {
                ::operator delete(p, std::align_val_t{csr_alignment});
            }
        };

        template <class T>
        using csr_buf = std::unique_ptr<T, csr_aligned_deleter>;

        template <class T>
        csr_buf<T> alloc_csr_buf(std::size_t n)
        {
            if (n == 0)
                return {nullptr};
            return {
                static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t{csr_alignment})), {}};
        }

    } // namespace detail

} // namespace spira
