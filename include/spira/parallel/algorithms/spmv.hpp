#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <spira/kernels/kernels.h>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/parallel/parallel_matrix.hpp>
#include <spira/traits.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // parallel spmv — y = A * x
    //
    // Each worker thread computes y[row_start..row_end) for its own partition
    // independently.  Output ranges are disjoint so no synchronisation is needed.
    //
    // Dispatch order (same as serial spmv):
    //   1. CSR flat-buffer scalar loop  (compact_preserve or compact_move)
    //   2. Per-row for_each_element     (no_compact fallback)
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::lock_policy LP, config::insert_policy IP,
              std::size_t SN>
    inline void spmv(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat,
                     const std::vector<V> &x, std::vector<V> &y)
    {
        if (x.size() != mat.n_cols())
            throw std::invalid_argument(
                "spmv: x size does not match matrix column count");
        if (y.size() != mat.n_rows())
            throw std::invalid_argument("spmv: y size does not match matrix row count");

        assert(mat.is_locked() && "parallel spmv: matrix must be locked");

        mat.execute([&x, &y](const auto &p, std::size_t)
                    {
    if (p.csr.is_built()) {
      const std::size_t *off = p.csr.offsets.get();

      if constexpr (std::is_same_v<L, layout::tags::soa_tag>) {
        const I *cols = p.csr.cols.get();
        const V *vals = p.csr.vals.get();
        for (std::size_t i = 0; i < p.size(); ++i) {
          V acc = traits::ValueTraits<V>::zero();
          for (std::size_t k = off[i]; k < off[i + 1]; ++k)
            acc += x[static_cast<std::size_t>(cols[k])] * vals[k];
          y[p.row_start + i] = acc;
        }
      } else // aos_tag: interleaved (col, val) pairs
      {
        const auto *pairs = p.csr.pairs.get();
        for (std::size_t i = 0; i < p.size(); ++i) {
          V acc = traits::ValueTraits<V>::zero();
          for (std::size_t k = off[i]; k < off[i + 1]; ++k)
            acc +=
                x[static_cast<std::size_t>(pairs[k].column)] * pairs[k].value;
          y[p.row_start + i] = acc;
        }
      }
    } else // no_compact fallback — iterate row buffers directly
    {
      for (std::size_t i = 0; i < p.rows.size(); ++i) {
        V acc = traits::ValueTraits<V>::zero();
        p.rows[i].for_each_element([&acc, &x](I col, V val) {
          acc += x[static_cast<std::size_t>(col)] * val;
        });
        y[p.row_start + i] = acc;
      }
    } });
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SIMD overload — soa_tag / uint32_t / float
    // ─────────────────────────────────────────────────────────────────────────────

    template <class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    inline void spmv(parallel_matrix<layout::tags::soa_tag, uint32_t, float, BT, BN,
                                     LP, IP, SN> &mat,
                     const std::vector<float> &x, std::vector<float> &y)
    {
        if (x.size() != mat.n_cols())
            throw std::invalid_argument(
                "spmv: x size does not match matrix column count");
        if (y.size() != mat.n_rows())
            throw std::invalid_argument("spmv: y size does not match matrix row count");

        assert(mat.is_locked() && "parallel spmv: matrix must be locked");

        mat.execute([&x, &y](const auto &p, std::size_t)
                    {
    if (p.csr.is_built()) {
      const uint32_t *cols = p.csr.cols.get();
      const float *vals = p.csr.vals.get();
      const std::size_t *off = p.csr.offsets.get();

      for (std::size_t i = 0; i < p.size(); ++i) {
        const std::size_t row_nnz = off[i + 1] - off[i];
        y[p.row_start + i] = kernel::sparse_dot_float(
            vals + off[i], cols + off[i], x.data(), row_nnz, x.size());
      }
    } else {
      for (std::size_t i = 0; i < p.rows.size(); ++i) {
        float acc = 0.0f;
        p.rows[i].for_each_element([&acc, &x](uint32_t col, float val) {
          acc += x[static_cast<std::size_t>(col)] * val;
        });
        y[p.row_start + i] = acc;
      }
    } });
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SIMD overload — soa_tag / uint32_t / double
    // ─────────────────────────────────────────────────────────────────────────────

    template <class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    inline void spmv(parallel_matrix<layout::tags::soa_tag, uint32_t, double, BT,
                                     BN, LP, IP, SN> &mat,
                     const std::vector<double> &x, std::vector<double> &y)
    {
        if (x.size() != mat.n_cols())
            throw std::invalid_argument(
                "spmv: x size does not match matrix column count");
        if (y.size() != mat.n_rows())
            throw std::invalid_argument("spmv: y size does not match matrix row count");

        assert(mat.is_locked() && "parallel spmv: matrix must be locked");

        mat.execute([&x, &y](const auto &p, std::size_t)
                    {
    if (p.csr.is_built()) {
      const uint32_t *cols = p.csr.cols.get();
      const double *vals = p.csr.vals.get();
      const std::size_t *off = p.csr.offsets.get();

      for (std::size_t i = 0; i < p.size(); ++i) {
        const std::size_t row_nnz = off[i + 1] - off[i];
        y[p.row_start + i] = kernel::sparse_dot_double(
            vals + off[i], cols + off[i], x.data(), row_nnz, x.size());
      }
    } else {
      for (std::size_t i = 0; i < p.rows.size(); ++i) {
        double acc = 0.0;
        p.rows[i].for_each_element([&acc, &x](uint32_t col, double val) {
          acc += x[static_cast<std::size_t>(col)] * val;
        });
        y[p.row_start + i] = acc;
      }
    } });
    }

} // namespace spira::parallel::algorithms
