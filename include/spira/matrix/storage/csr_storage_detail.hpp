#pragma once

#include <algorithm>
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

        // Row slot table shared by both layouts.
        //
        // Row i owns the slots [row_start[i], row_start[i] + row_cap[i]) of the
        // data arrays and uses the first row_len[i] of them; the rest is slack
        // that lets the row grow in place. Rows are not required to be stored
        // in index order: a row that outgrows its slot is moved to the free
        // tail [end, capacity), and the slot it leaves behind is counted in
        // holes until the next repack. row_grown[i] records that row i has
        // outgrown a slot, which decides whether it gets slack when the CSR
        // is laid out again (see config::row_slack).
        struct csr_row_table
        {
            std::size_t n_rows{0};
            std::size_t nnz{0};      // live entries: sum of row_len
            std::size_t capacity{0}; // allocated slots in the data arrays
            std::size_t end{0};      // slots [0, end) are assigned to rows
            std::size_t holes{0};    // slots in [0, end) no row owns any more

            std::unique_ptr<std::size_t[]> row_start;
            std::unique_ptr<std::size_t[]> row_len;
            std::unique_ptr<std::size_t[]> row_cap;
            std::unique_ptr<bool[]> row_grown;

            csr_row_table() = default;

            csr_row_table(std::size_t n_rows_, std::size_t capacity_)
                : n_rows{n_rows_}, capacity{capacity_},
                  row_start{std::make_unique<std::size_t[]>(n_rows_)},
                  row_len{std::make_unique<std::size_t[]>(n_rows_)},
                  row_cap{std::make_unique<std::size_t[]>(n_rows_)},
                  row_grown{std::make_unique<bool[]>(n_rows_)}
            {
            }

            // Copies the table only; the owning storage copies the data slots.
            csr_row_table(const csr_row_table &other)
                : n_rows{other.n_rows}, nnz{other.nnz}, capacity{other.end},
                  end{other.end}, holes{other.holes},
                  row_start{copy_array(other.row_start, other.n_rows)},
                  row_len{copy_array(other.row_len, other.n_rows)},
                  row_cap{copy_array(other.row_cap, other.n_rows)},
                  row_grown{copy_array(other.row_grown, other.n_rows)}
            {
            }

            csr_row_table &operator=(const csr_row_table &) = delete;
            csr_row_table(csr_row_table &&) = default;
            csr_row_table &operator=(csr_row_table &&) = default;

            [[nodiscard]] bool is_built() const noexcept { return row_start != nullptr; }

        private:
            template <class T>
            static std::unique_ptr<T[]> copy_array(const std::unique_ptr<T[]> &src, std::size_t n)
            {
                if (!src)
                    return nullptr;
                auto dst = std::make_unique<T[]>(n);
                std::copy_n(src.get(), n, dst.get());
                return dst;
            }
        };

    } // namespace detail

} // namespace spira
