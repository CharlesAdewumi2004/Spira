#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>

#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{

    // Helper: extract key from iterator element regardless of AoS/SoA proxy type.
    namespace detail {
        auto key_of(const auto& entry) -> decltype(auto) {
            if constexpr (requires { entry.first_ref(); })
                return entry.first_ref();
            else
                return entry.first;
        }
        auto val_of(const auto& entry) -> decltype(auto) {
            if constexpr (requires { entry.second_ref(); })
                return entry.second_ref();
            else
                return entry.second;
        }
    }

    /// Merge two locked, sorted rows into a single open output row.
    /// A and B must be locked (sorted slabs). out must be in open mode and empty.
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    void addRows(const spira::row<Layout, I, V> &A, const spira::row<Layout, I, V> &B, spira::row<Layout, I, V> &out)
    {
        assert(A.is_locked() && "addRows: row A must be locked");
        assert(B.is_locked() && "addRows: row B must be locked");

        out.clear();

        auto itA = A.begin();
        auto itB = B.begin();
        const auto endA = A.end();
        const auto endB = B.end();

        while (itA != endA && itB != endB)
        {
            auto ae = *itA;
            auto be = *itB;
            const auto a_col = detail::key_of(ae);
            const auto b_col = detail::key_of(be);
            const auto a_val = detail::val_of(ae);
            const auto b_val = detail::val_of(be);

            if (a_col == b_col)
            {
                V sum = a_val + b_val;
                if (!spira::traits::ValueTraits<V>::is_zero(sum))
                {
                    out.insert(a_col, sum);
                }
                ++itA;
                ++itB;
            }
            else if (a_col < b_col)
            {
                out.insert(a_col, a_val);
                ++itA;
            }
            else
            {
                out.insert(b_col, b_val);
                ++itB;
            }
        }

        while (itA != endA)
        {
            auto ae = *itA++;
            out.insert(detail::key_of(ae), detail::val_of(ae));
        }

        while (itB != endB)
        {
            auto be = *itB++;
            out.insert(detail::key_of(be), detail::val_of(be));
        }
    }

    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    spira::matrix<Layout, I, V> MatrixAddition(const spira::matrix<Layout, I, V> &A, const spira::matrix<Layout, I, V> &B)
    {
        if (A.shape() != B.shape())
        {
            throw std::invalid_argument("Matrices aren't the same size");
        }

        assert(A.is_locked() && "MatrixAddition: A must be locked");
        assert(B.is_locked() && "MatrixAddition: B must be locked");

        const auto [r, c] = A.shape();
        spira::matrix<Layout, I, V> out(r, c);

        for (std::size_t i = 0; i < A.n_rows(); ++i)
        {
            const I ri = static_cast<I>(i);
            addRows(A.row_at(ri), B.row_at(ri), out.row_at_mut(ri));
        }

        out.lock();
        return out;
    }

} // namespace spira::algorithms
