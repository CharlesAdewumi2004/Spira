#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spira/matrix/matrix.hpp>
#include <spira/traits.hpp>

namespace spira::serial::algorithms
{

    /// Merge two locked rows into a single open output row.
    /// A and B must be locked. out must be in open mode and empty.
    /// Iteration is via for_each_element, which reads the CSR slice.
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V, class BT, std::size_t BN>
    void addRows(const spira::row<Layout, I, V, BT, BN> &A, const spira::row<Layout, I, V, BT, BN> &B,
                 spira::row<Layout, I, V, BT, BN> &out)
    {
        if (!A.is_locked())
            throw std::logic_error("addRows: row A must be locked");
        if (!B.is_locked())
            throw std::logic_error("addRows: row B must be locked");

        out.clear();

        // Collect A and B entries into flat arrays for two-pointer merge.
        // Thread-local vectors avoid a heap allocation per row after the first call.
        thread_local std::vector<std::pair<I, V>> a_entries, b_entries;
        a_entries.clear();
        b_entries.clear();
        a_entries.reserve(A.size());
        b_entries.reserve(B.size());

        A.for_each_element([](const I &col, const V &val) {
            a_entries.push_back({col, val});
        });
        B.for_each_element([](const I &col, const V &val) {
            b_entries.push_back({col, val});
        });

        std::size_t ai = 0, bi = 0;
        while (ai < a_entries.size() && bi < b_entries.size())
        {
            const auto a_col = a_entries[ai].first;
            const auto b_col = b_entries[bi].first;
            if (a_col == b_col)
            {
                V sum = a_entries[ai].second + b_entries[bi].second;
                if (!spira::traits::ValueTraits<V>::is_zero(sum))
                    out.insert(a_col, sum);
                ++ai; ++bi;
            }
            else if (a_col < b_col)
            {
                out.insert(a_col, a_entries[ai].second);
                ++ai;
            }
            else
            {
                out.insert(b_col, b_entries[bi].second);
                ++bi;
            }
        }
        while (ai < a_entries.size()) { out.insert(a_entries[ai].first, a_entries[ai].second); ++ai; }
        while (bi < b_entries.size()) { out.insert(b_entries[bi].first, b_entries[bi].second); ++bi; }
    }

    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V, class BT, std::size_t BN>
    spira::matrix<Layout, I, V, BT, BN> MatrixAddition(const spira::matrix<Layout, I, V, BT, BN> &A, const spira::matrix<Layout, I, V, BT, BN> &B)
    {
        if (A.shape() != B.shape())
        {
            throw std::invalid_argument("Matrices aren't the same size");
        }

        if (!A.is_locked())
            throw std::logic_error("MatrixAddition: A must be locked");
        if (!B.is_locked())
            throw std::logic_error("MatrixAddition: B must be locked");

        const auto [r, c] = A.shape();
        spira::matrix<Layout, I, V, BT, BN> out(r, c);

        for (std::size_t i = 0; i < A.n_rows(); ++i)
        {
            const I ri = static_cast<I>(i);
            addRows(A.row_at(ri), B.row_at(ri), out.row_at_mut(ri));
        }

        out.lock();
        return out;
    }

} // namespace spira::serial::algorithms
