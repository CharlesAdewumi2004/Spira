#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <new>

namespace spira
{

    inline constexpr std::size_t csr_alignment = 64;

    // ─────────────────────────────────────────────────────────────────────────────
    // csr_storage<I, V>
    //
    // Three flat arrays representing a Compressed Sparse Row matrix:
    //
    //   offsets[i]     — index into cols/vals where row i begins
    //   offsets[n_rows] — total nnz (one past the end of the last row)
    //   cols[k]        — column index of the k-th stored element
    //   vals[k]        — value of the k-th stored element
    //
    // cols and vals are 64-byte aligned for SIMD access.
    // offsets does not require SIMD alignment.
    //
    // This struct is a plain data container. It knows nothing about lock_policy,
    // row buffers, or matrix operations. Call is_built() to check whether the
    // arrays have been allocated.
    // ─────────────────────────────────────────────────────────────────────────────

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
                return {nullptr, {}};
            return {
                static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t{csr_alignment})),
                {}};
        }

    } // namespace detail

    template <class I, class V>
    struct csr_storage
    {
        std::size_t n_rows{0};
        std::size_t nnz{0};

        std::unique_ptr<std::size_t[]> offsets; // [n_rows + 1]
        detail::csr_buf<I> cols;                // [nnz], 64-byte aligned
        detail::csr_buf<V> vals;                // [nnz], 64-byte aligned

        csr_storage() = default;

        csr_storage(std::size_t n_rows_, std::size_t nnz_)
            : n_rows{n_rows_}, nnz{nnz_}, offsets{std::make_unique<std::size_t[]>(n_rows_ + 1)}, cols{detail::alloc_csr_buf<I>(nnz_)}, vals{detail::alloc_csr_buf<V>(nnz_)}
        {
        }

        csr_storage(const csr_storage &) = delete;
        csr_storage &operator=(const csr_storage &) = delete;
        csr_storage(csr_storage &&) = default;
        csr_storage &operator=(csr_storage &&) = default;

        /// True if the three arrays have been allocated (i.e. build_csr() has run).
        [[nodiscard]] bool is_built() const noexcept { return offsets != nullptr; }
    };

} // namespace spira
