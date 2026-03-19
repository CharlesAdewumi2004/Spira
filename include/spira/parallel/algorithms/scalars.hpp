#pragma once

#include <stdexcept>

#include <spira/traits.hpp>
#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // multiplication_scaler — in-place: mat must be open.
    // Each worker scales committed CSR entries in its own partition rows.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    void multiplication_scaler(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat, V scaler)
    {
        if (!mat.is_open())
            throw std::logic_error("multiplication_scaler: matrix must be open");

        mat.execute([scaler](auto &p, std::size_t)
        {
            for (auto &r : p.rows)
                r.for_each_committed_element([&r, scaler](I col, V val)
                { r.insert(col, val * scaler); });
        });
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // multiplication_scaler — copy: mat locked → out open, scaled, then locked.
    // A and out must have the same shape and thread count.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    void multiplication_scaler(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat,
                               parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &out,
                               V scaler)
    {
        if (!mat.is_locked())
            throw std::logic_error("multiplication_scaler: input matrix must be locked");
        if (!out.is_open())
            throw std::logic_error("multiplication_scaler: output matrix must be open");
        if (mat.shape() != out.shape())
            throw std::invalid_argument("multiplication_scaler: shape mismatch");
        if (mat.n_threads() != out.n_threads())
            throw std::invalid_argument("multiplication_scaler: thread count mismatch");

        mat.execute([&out, scaler](const auto &p_in, std::size_t t)
        {
            auto &p_out = out.partition_at(t);
            for (std::size_t i = 0; i < p_in.rows.size(); ++i)
                p_in.rows[i].for_each_committed_element(
                    [&p_out, i, scaler](I col, V val)
                    { p_out.rows[i].insert(col, val * scaler); });
        });

        out.lock();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // division_scaler — in-place
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    void division_scaler(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat, V scaler)
    {
        if (traits::ValueTraits<V>::is_zero(scaler))
            throw std::domain_error("division by zero");
        if (!mat.is_open())
            throw std::logic_error("division_scaler: matrix must be open");

        mat.execute([scaler](auto &p, std::size_t)
        {
            for (auto &r : p.rows)
                r.for_each_committed_element([&r, scaler](I col, V val)
                { r.insert(col, val / scaler); });
        });
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // division_scaler — copy
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    void division_scaler(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat,
                         parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &out,
                         V scaler)
    {
        if (traits::ValueTraits<V>::is_zero(scaler))
            throw std::domain_error("division by zero");
        if (!mat.is_locked())
            throw std::logic_error("division_scaler: input matrix must be locked");
        if (!out.is_open())
            throw std::logic_error("division_scaler: output matrix must be open");
        if (mat.shape() != out.shape())
            throw std::invalid_argument("division_scaler: shape mismatch");
        if (mat.n_threads() != out.n_threads())
            throw std::invalid_argument("division_scaler: thread count mismatch");

        mat.execute([&out, scaler](const auto &p_in, std::size_t t)
        {
            auto &p_out = out.partition_at(t);
            for (std::size_t i = 0; i < p_in.rows.size(); ++i)
                p_in.rows[i].for_each_committed_element(
                    [&p_out, i, scaler](I col, V val)
                    { p_out.rows[i].insert(col, val / scaler); });
        });

        out.lock();
    }

} // namespace spira::parallel::algorithms
