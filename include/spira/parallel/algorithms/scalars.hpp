#pragma once

#include <stdexcept>
#include <string>

#include <spira/traits.hpp>
#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    namespace detail
    {
        // In place: commit pending edits so every current value is in the CSR,
        // reopen, then stage each scaled value into its row. Reading a row's
        // CSR slice while writing to its buffer is safe; they are separate.
        template <class M, class Op>
        void scale_in_place(M &mat, Op op)
        {
            mat.lock();
            mat.open();
            mat.execute([&op](auto &p, std::size_t)
            {
                for (std::size_t i = 0; i < p.rows.size(); ++i)
                    p.rows[i].for_each_element([&p, &op, i](auto col, auto val)
                    { p.writable_row(i).insert(col, op(val)); });
            });
        }

        // Copy: out (open, same shape and thread count) receives op(value) for
        // every entry of the locked matrix mat, then is locked.
        template <class M, class Op>
        void scale_copy(M &mat, M &out, Op op, const char *name)
        {
            if (!mat.is_locked())
                throw std::logic_error(std::string(name) + ": input matrix must be locked");
            if (!out.is_open())
                throw std::logic_error(std::string(name) + ": output matrix must be open");
            if (mat.shape() != out.shape())
                throw std::invalid_argument(std::string(name) + ": shape mismatch");
            if (mat.n_threads() != out.n_threads())
                throw std::invalid_argument(std::string(name) + ": thread count mismatch");

            mat.execute([&out, &op](const auto &p_in, std::size_t t)
            {
                auto &p_out = out.partition_at(t);
                for (std::size_t i = 0; i < p_in.rows.size(); ++i)
                    p_in.rows[i].for_each_element([&p_out, &op, i](auto col, auto val)
                    { p_out.writable_row(i).insert(col, op(val)); });
            });
            out.lock();
        }
    } // namespace detail

    /// In-place scalar multiply. mat must be open and stays open; the scaled
    /// values are committed by the next lock().
    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN,
              config::insert_policy IP, std::size_t SN>
    void multiplication_scaler(parallel_matrix<L, I, V, BT, BN, IP, SN> &mat, V scaler)
    {
        if (!mat.is_open())
            throw std::logic_error("multiplication_scaler: matrix must be open");
        detail::scale_in_place(mat, [scaler](V v) { return v * scaler; });
    }

    /// Copy: mat locked, out open with the same shape and thread count; out is
    /// left locked.
    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN,
              config::insert_policy IP, std::size_t SN>
    void multiplication_scaler(parallel_matrix<L, I, V, BT, BN, IP, SN> &mat,
                               parallel_matrix<L, I, V, BT, BN, IP, SN> &out,
                               V scaler)
    {
        detail::scale_copy(mat, out, [scaler](V v) { return v * scaler; }, "multiplication_scaler");
    }

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN,
              config::insert_policy IP, std::size_t SN>
    void division_scaler(parallel_matrix<L, I, V, BT, BN, IP, SN> &mat, V scaler)
    {
        if (traits::ValueTraits<V>::is_zero(scaler))
            throw std::domain_error("division by zero");
        if (!mat.is_open())
            throw std::logic_error("division_scaler: matrix must be open");
        detail::scale_in_place(mat, [scaler](V v) { return v / scaler; });
    }

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN,
              config::insert_policy IP, std::size_t SN>
    void division_scaler(parallel_matrix<L, I, V, BT, BN, IP, SN> &mat,
                         parallel_matrix<L, I, V, BT, BN, IP, SN> &out,
                         V scaler)
    {
        if (traits::ValueTraits<V>::is_zero(scaler))
            throw std::domain_error("division by zero");
        detail::scale_copy(mat, out, [scaler](V v) { return v / scaler; }, "division_scaler");
    }

} // namespace spira::parallel::algorithms
