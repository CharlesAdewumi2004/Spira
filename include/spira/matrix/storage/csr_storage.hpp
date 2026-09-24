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
// Both variants share a row slot table (detail::csr_row_table): row i lives
// in [row_start[i], row_start[i] + row_len[i]) and owns row_cap[i] slots, so
// it has slack to grow in place. Readers must use row_start/row_len; rows are
// not necessarily stored in index order, and the gaps hold stale data.
// The layout policy only affects the locked (read) structure; open-mode
// buffering is always a plain growable array of (col, val) pairs.
//
// csr_slice<LayoutTag, I, V>
//
// A non-owning view into one row's slice of a csr_storage.
// Installed on each row by matrix::lock() via row::set_csr_slice(), built
// with csr_storage::slice(row).
//
// Provides: is_set(), reset(), binary_search(col), for_each(fn), accumulate().
// The layout tag selects the underlying pointer type and access pattern.
// ─────────────────────────────────────────────────────────────────────────────

#include <spira/matrix/storage/csr_storage_soa.hpp>
#include <spira/matrix/storage/csr_storage_aos.hpp>
