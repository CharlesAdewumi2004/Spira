#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// csr_storage<LayoutTag, I, V>
//
// The layout tag determines the flat memory arrangement of the locked CSR:
//
//   soa_tag — two separate flat arrays: cols[nnz] and vals[nnz].
//             SIMD SpMV loads values and column indices independently
//             with single aligned vector loads.
//
//   aos_tag — one interleaved array: elementPair<I,V> pairs[nnz].
//             SIMD SpMV must gather/stride over interleaved data.
//
// Both variants share: offsets[n_rows+1] for row boundaries.
// The layout policy only affects the locked (read) structure; open-mode
// buffering is always a plain growable array of (col, val) pairs.
//
// csr_slice<LayoutTag, I, V>
//
// A non-owning view into one row's slice of a csr_storage.
// Installed on each row by matrix::lock() via row::set_csr_slice().
//
// Provides: is_set(), reset(), binary_search(col), for_each(fn), accumulate().
// The layout tag selects the underlying pointer type and access pattern.
// ─────────────────────────────────────────────────────────────────────────────

#include <spira/matrix/storage/csr_storage_soa.hpp>
#include <spira/matrix/storage/csr_storage_aos.hpp>
