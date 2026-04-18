# Architecture

This document walks through the library layer by layer. It assumes you have
read [overview.md](overview.md) and know the four stages. Every layer has a
PlantUML diagram in [diagrams/](diagrams/) that you can render alongside the
text. File paths below are relative to the project root.

## Table of contents

1. [Layer stack and dependency direction](#1-layer-stack-and-dependency-direction)
2. [Concepts and traits](#2-concepts-and-traits)
3. [Layout tags](#3-layout-tags)
4. [Buffers](#4-buffers)
5. [Row](#5-row)
6. [Compressed storage](#6-compressed-storage)
7. [Matrix](#7-matrix)
8. [The open and locked lifecycle](#8-the-open-and-locked-lifecycle)
9. [Serial algorithms](#9-serial-algorithms)
10. [SIMD kernels and runtime dispatch](#10-simd-kernels-and-runtime-dispatch)
11. [Thread pool](#11-thread-pool)
12. [Partition](#12-partition)
13. [Parallel matrix](#13-parallel-matrix)
14. [Parallel algorithms](#14-parallel-algorithms)
15. [Insert staging policy](#15-insert-staging-policy)
16. [Putting it together: a full SpMV trace](#16-putting-it-together-a-full-spmv-trace)

## 1. Layer stack and dependency direction

The library has a clean bottom up dependency order. Nothing below looks up
at anything above. See [05-component-layers.puml](diagrams/05-component-layers.puml)
for the full picture.

```
    parallel/algorithms/*
           |
    parallel_matrix, partition, thread_pool
           |
    serial/*  (spmv, spgemm, matrix_add, transpose, accumulate, scalars)
           |
    matrix, matrix_operators
           |
    row
           |
    storage/*  (csr_storage_*oa, csr_build)
           |
    buffer/*   (aos_array_buffer, soa_array_buffer, hash_map_buffer)
           |
    layout tags, element_pair
           |
    concepts, traits, config
```

The SIMD kernel library (`src/kernels/`) sits off to the side. It is linked
as a static library and is pulled in by the serial SpMV specialisation for
`soa_tag + uint32_t + float/double`. The parallel SpMV calls the same
specialisation from each worker thread, so both paths share the same kernel
function pointers.

## 2. Concepts and traits

The two foundational concepts are defined in
[concepts.hpp](../include/spira/concepts.hpp).

`concepts::Indexable` constrains the column index type. It requires an
unsigned integral of at least 32 bits and explicitly rejects `bool` and
character types. This rules out 16 bit indices and makes the SIMD kernels
simpler because they can assume `uint32_t` or `uint64_t`.

`concepts::Valueable` constrains the value type. It requires arithmetic
operations (`+`, `*`, `+=`, `*=`), default construction, move and copy
construction, equality, and a `traits::ValueTraits<V>::zero()` sentinel with
an `is_zero` test. Floating point types, integers, and `std::complex<T>`
all satisfy it; raw `bool` and character types do not.

`traits::ValueTraits<V>` is the scalar vocabulary used everywhere the
library needs a "zero" or needs to know what type to accumulate into. For
integer `V` the accumulation type is widened to at least 64 bits so that
dot products of long rows cannot overflow. For floating point the
accumulation type is the same as the value type. For `std::complex<T>` the
zero is `complex(0, 0)` and `is_zero` is a tolerance aware magnitude
check. All of this is in [traits.hpp](../include/spira/traits.hpp).

`config.hpp` defines the three enumerations that control lifecycle
behaviour: `matrix_mode` (open, locked), `lock_policy`
(no_compact, compact_preserve, compact_move), and `insert_policy` (direct,
staged). It also pins the search policy used inside buffers to a hybrid
binary-linear search from the `boundcraft` dependency.

## 3. Layout tags

There are exactly two layout tags, defined in
[layout_tags.hpp](../include/spira/matrix/layout/layout_tags.hpp).

```cpp
namespace spira::layout::tags {
    struct aos_tag {};
    struct soa_tag {};
}
```

These tags are dispatch keys and carry no runtime state. The `ValidLayoutTag`
concept accepts only these two types. The rest of the library uses them to
pick specialisations of the CSR storage, the buffers, and the SpMV
algorithm.

`element_pair.hpp` defines the `elementPair<I, V>` struct that the AoS path
uses to keep a column and its value together in one record. It is trivially
copyable and the two fields sit in memory in `{column, value}` order.

See [11-memory-layout.puml](diagrams/11-memory-layout.puml) for the side by
side picture of AoS vs SoA CSR.

## 4. Buffers

A buffer is a growable, unsorted staging area for inserts into a single
row. The interface is defined by a CRTP base in
[buffer_base.hpp](../include/spira/matrix/buffer/buffer_base.hpp) and
enforced with the `Buffer` concept.

```cpp
template <class Derived, class I, class V>
class base_buffer {
    bool empty() const;
    size_t size() const;
    void push_back(I col, V val);
    bool contains(I col) const;
    const V *get_ptr(I col) const;
    V accumulate() const;
    void sort_and_dedup();
};
```

There are three concrete buffers. See
[09-buffer-hierarchy.puml](diagrams/09-buffer-hierarchy.puml) for the class
diagram.

**`aos_array_buffer<I, V, N>`** stores `elementPair<I, V>` in a `std::vector`
and keeps an `ankerl::unordered_dense::map<I, size_t>` that tracks the
latest insert index for each column. This map is what makes open mode reads
O(1) and gives the buffer last write wins semantics: if the same column is
inserted twice the map is updated to point at the newer entry.

**`soa_array_buffer<I, V, N>`** stores columns and values in two parallel
vectors with the same last write wins index map. It is the layout used by
the SoA SpMV kernel because the column and value arrays can be loaded as
independent SIMD vectors without any gather.

**`hash_map_buffer<I, V>`** uses only the `ankerl::unordered_dense::map`
directly without a backing vector. It is the fastest buffer for pathological
insertion patterns with heavy duplication because `push_back` is a single
map update, but it has no ordering so `sort_and_dedup()` has to iterate the
map and materialise a sorted array.

Each buffer exposes a `sort_and_dedup()` method that is called by
`row::lock()`. The method sorts entries by column, resolves duplicates
using the index map, and filters out values that `ValueTraits::is_zero`
returns true for. Filtered zeros are important because they keep the CSR
nnz count accurate after a user writes a zero to overwrite an existing
non zero.

The buffer tag traits in
[buffer_tag_traits.hpp](../include/spira/matrix/buffer/buffer_tag_traits.hpp)
resolve a tag plus index plus value plus initial size into a concrete
buffer type. This is the mechanism the matrix uses to plumb the
`BufferTag` template parameter down to the row.

## 5. Row

A `row<LayoutTag, I, V, BufferTag, BufferN>` is the per matrix row unit of
storage. Its definition is in
[row.hpp](../include/spira/matrix/row.hpp).

A row has three pieces of state:

1. A buffer (one of the three above) for unsorted inserts.
2. A `csr_slice<LayoutTag, I, V>` which is a non owning view into the flat
   CSR array. The slice is empty when the matrix has not been locked yet.
3. A mode flag (open or locked).

Inserts in open mode go straight to the buffer. Reads check the buffer's
index map first (O(1) hit) and fall back to the slice (binary search in
locked mode, empty check otherwise). A locked row treats the buffer as
empty; the slice is the authoritative store.

The row does not own its CSR data. The `csr_slice` holds three pointers
(or two, for AoS) into a flat block that is owned by the matrix. This is
what makes locked mode reads zero allocation: you go through a pointer
into an already built array.

When `matrix::lock()` runs it calls `row::lock()` on each row, which does
two things. First it calls `buffer.sort_and_dedup()`. Second it sets the
row mode to locked. The actual CSR array is built by a separate function
in the storage layer (see next section) and the slice is installed
afterwards by the matrix via `row::set_csr_slice()`.

## 6. Compressed storage

Compressed Sparse Row storage is the on disk format for a locked matrix.
The storage types live under
[include/spira/matrix/storage/](../include/spira/matrix/storage/).

`csr_storage<LayoutTag, I, V>` has two specialisations, one per layout.

The **SoA** specialisation in
[csr_storage_soa.hpp](../include/spira/matrix/storage/csr_storage_soa.hpp)
has three arrays: `offsets[n_rows+1]`, `cols[nnz]`, `vals[nnz]`. All three
are allocated with `std::aligned_alloc` at 64 byte alignment (one cache
line). The `offsets` array has one entry per row plus a sentinel, and row
`i` occupies the range `cols[offsets[i]..offsets[i+1])` and
`vals[offsets[i]..offsets[i+1])`.

The **AoS** specialisation in
[csr_storage_aos.hpp](../include/spira/matrix/storage/csr_storage_aos.hpp)
keeps a single `pairs[nnz]` array of `elementPair<I, V>` instead of
separate cols and vals. This gives one cache line carrying both the
column and its value, which is the right layout for scalar per element
access patterns that touch both at once.

`csr_slice<LayoutTag, I, V>` is the non owning row view. It stores a
pointer (or two) into the main arrays, a nnz count, and supports
`binary_search(col)`, `for_each_element(f)`, and `accumulate()`. All the
per row read operations go through a slice.

`csr_build.hpp` has two free functions. `build_csr(rows)` does a first
time build: one prefix sum pass over row sizes to fill `offsets`, then
one copy pass to populate the cols and vals arrays. Both passes are
O(nnz).

`merge_csr(rows, old_csr, dirty)` is the incremental version. It does a
two pointer merge of the existing CSR array with the new buffer entries,
on a per row basis. Clean rows (not dirty) short circuit to a
`memmove` of the old slice into the new allocation. Dirty rows do the
full merge. If the total upper bound fits into the old capacity the
merge reuses the existing allocation with a reverse order in place
merge; otherwise a new allocation is made at 1.5x the needed size and a
forward merge is used.

See [07-csr-build-activity.puml](diagrams/07-csr-build-activity.puml) for
the activity diagram of this logic.

## 7. Matrix

`spira::matrix<LayoutTag, I, V, BufferTag, BufferN, LP>` in
[matrix.hpp](../include/spira/matrix/matrix.hpp) is the single threaded
entry point. It is copyable and movable, and owns:

- `std::vector<row_type> rows_`: one row per matrix row.
- `csr_storage<LayoutTag, I, V> csr_`: the flat CSR, only populated after
  the first lock.
- `std::vector<bool> dirty_`: bit per row, set when `insert` is called in
  open mode, cleared by `lock`.
- `config::matrix_mode mode_`: the current open or locked state.

The public API is divided into shape queries (`shape`, `n_rows`, `n_cols`),
mode transitions (`lock`, `open`, `is_locked`, `is_open`), per element
operations (`insert`, `get`, `contains`, `row_nnz`), arithmetic
(`operator+`, `operator*`, `operator~`), and iteration
(`for_each_row`, `for_each_nnz_row`).

The matrix also exposes `load_csr(csr_storage&&)` which is the escape
hatch for constructing a locked matrix from an already built CSR array,
skipping the lock pipeline entirely. This is useful when you read a
matrix from disk or receive it from another library.

See [01-class-diagram.puml](diagrams/01-class-diagram.puml) for the
complete class diagram of the matrix and its neighbours.

## 8. The open and locked lifecycle

The lifecycle has two stable states and a transition step in each direction.
The state diagram is in
[02-state-lifecycle.puml](diagrams/02-state-lifecycle.puml) and the lock
transition sequence is in
[03-lock-sequence.puml](diagrams/03-lock-sequence.puml).

### Open mode

In open mode the matrix is mutable. `insert(r, c, v)` performs three
operations: it writes to `rows_[r].buffer`, updates the buffer's column
index map, and sets `dirty_[r] = true`. All three are O(1) amortised.

Reads in open mode consult the buffer's index map first. If the column is
in the map the read returns in O(1). If it is not, and the matrix has
been locked previously, the read falls back to a binary search in the
row's CSR slice. This is the only scenario where an open mode read can
hit the slice.

### Locking

`matrix::lock()` runs a four step pipeline.

1. **Sort and dedup each row buffer.** This is the per row
   `buffer.sort_and_dedup()` call described earlier. It is the
   dominant cost for large buffers.
2. **Build or merge the CSR.** If `csr_` is empty this is a
   `build_csr`; otherwise it is a `merge_csr` with the dirty bitset.
3. **Install CSR slices.** Every row gets a `csr_slice` pointing at its
   position in the new CSR array.
4. **Clear buffers.** If the lock policy is `compact_move` the buffers
   are deallocated; otherwise they are emptied but keep their capacity.

After these steps `mode_` flips to locked and the dirty bitset is
cleared.

### Locked mode

Locked mode is a read only view. `insert` throws; `get`, `contains`,
`row_nnz`, `accumulate`, SpMV, SpGEMM, and the iteration helpers all
work and go through the CSR slices without touching buffers.

### Re-opening

`matrix::open()` sets `mode_` back to open. The CSR array is not touched.
Under `compact_preserve` the row buffers still exist and inserts work
immediately. Under `compact_move` the row buffers are re-allocated here,
which is the one linear cost of re-opening.

### The dirty bitset

The dirty bitset is the reason re-locking a matrix is cheap when only a
few rows changed. After the first lock, subsequent locks look like:

```
for each row i:
    if dirty_[i]:
        merge_csr_row(old_slice[i], new_buffer[i]) -> new_csr
    else:
        memcpy(old_slice[i], new_csr)
```

For a matrix with 10,000 rows and 256 inserts per re-lock cycle the
expected number of dirty rows is approximately 256 (usually fewer due to
collisions), which is 2.5% of the matrix. The remaining 97.5% of rows
hit the `memcpy` fast path and contribute nothing more than a sequential
memory copy to the re-lock cost.

## 9. Serial algorithms

The single threaded algorithms live in
[include/spira/serial/](../include/spira/serial/) and are all free
functions in the `spira::serial::algorithms` namespace.

| Function | File | What it does |
|----------|------|--------------|
| `spmv(A, x, y)` | `spmv.hpp` | `y = A * x` for a locked matrix |
| `spgemm(A, B)` | `spgemm.hpp` | Returns `C = A * B`, a new locked matrix |
| `transpose(A)` | `transpose.hpp` | Returns `A^T`, a new locked matrix |
| `matrix_add(A, B)` | `matrix_addition.hpp` | Returns `A + B` |
| `accumulate(A, r)` | `accumulate.hpp` | Sum of row `r` |
| `scale(A, s)` | `scalars.hpp` | Returns `s * A` |

Each function takes matrices by const reference and produces a new output
matrix where applicable. They all assume a locked input. If you pass an
open matrix you get an exception.

SpMV has a SIMD specialisation that only kicks in when the layout is SoA,
the index type is `uint32_t`, and the value type is `float` or `double`.
Everything else falls through to a per row `for_each_element` loop that
uses the CSR slice directly.

`matrix_operators.hpp` defines `operator+`, `operator-`, `operator*`,
`operator/`, `operator~`, and their compound assignment variants. Each
operator dispatches to the corresponding algorithm function, so
`C = A + B` is just sugar for `matrix_add(A, B)`.

## 10. SIMD kernels and runtime dispatch

The kernel library is in [src/kernels/](../src/kernels/). It is the only
part of the project that is not header only, and it produces a static
library called `spira_kernels`.

### What the kernels compute

All the kernels compute the sparse dot product of one row of a CSR matrix
against a dense vector:

```
sparse_dot(vals, cols, x, n, x_size) = sum over k in [0, n):
                                          vals[k] * x[cols[k]]
```

`vals` and `cols` are sequential, because that is how CSR lays them out.
`x[cols[k]]` is a scalar gather because the column indices are irregular.
Every kernel (scalar, SSE, AVX2, AVX-512, NEON) implements exactly this
contract, just with a different number of elements per iteration.

### The dispatch layer

`dispatch.cpp` contains a single static initialiser that runs once per
process at load time. Its job is to:

1. Call `RuntimeConfig::get()` which probes the CPU via CPUID (on x86),
   `getauxval(AT_HWCAP)` or `sysctlbyname` (on ARM), and measures DRAM
   latency with a pointer chase chain.
2. Pick the best available kernel for `sparse_dot_double` and
   `sparse_dot_float` based on the CPU feature set.
3. Assign the chosen function pointers to two global symbols.

The priority order, from best to worst, is AVX-512, AVX2 with FMA,
SSE 4.2, NEON, scalar. On an AVX2 x86 box you end up with
`sparse_dot_double = &sparse_dot_double_avx`; on an Apple M series box
you end up with `sparse_dot_double = &sparse_dot_double_neon`.

The SpMV specialisation calls the function pointer directly, with no
dispatch branch on the hot path. See
[08-kernel-dispatch.puml](diagrams/08-kernel-dispatch.puml) for the
startup flow.

### Kernel internals

Each kernel unrolls the inner loop to feed enough FMA units to hide the
latency of the gather. The AVX2 double kernel in
`src/kernels/dot_impls/x86/dot_avx.cpp` uses four independent
accumulators, each 4-wide, so it works on 16 elements per unrolled
iteration and retires one FMA chain per cycle on Haswell and later.
The AVX-512 variant does the same with 8-wide vectors.

The gather is always scalar because there is no hardware gather
instruction on x86 that is faster than four scalar loads when the
indices are unpredictable, which is the normal case for sparse
matrices. Writing the gather by hand also avoids alignment faults when
the target addresses straddle cache lines.

The tail of every kernel handles the last `n mod W` elements with a
scalar loop (where `W` is the SIMD width). This keeps the main loop
branchless.

### Hardware detection details

`hw_detect.hpp` defines a `CpuFeatures` struct with a bool per feature
(`sse2`, `sse42`, `avx`, `avx2`, `fma`, `avx512f`, `avx512vl`, `avx512bw`,
`avx512dq`, `neon`). On x86 it queries CPUID leaves 1 and 7 and checks
the XSAVE bits to make sure the OS actually saves the YMM and ZMM
register files. On Linux ARM it reads `/proc/cpuinfo` and the auxiliary
vector. On Apple it uses `sysctlbyname`. On Windows it uses
`IsProcessorFeaturePresent`.

`runtime_config.hpp` wraps `CpuFeatures` and the measured DRAM latency
into a singleton that can be queried from the dispatch code.

## 11. Thread pool

`spira::parallel::thread_pool` in
[thread_pool.hpp](../include/spira/parallel/thread_pool.hpp) is a
fixed size pool of `std::jthread` workers. See
[10-thread-pool.puml](diagrams/10-thread-pool.puml) for the sequence
diagram.

The pool has one entry point: `execute(fn)`. It broadcasts the function
object to all workers, each calls `fn(thread_id)`, and the call blocks
until all workers have returned.

The synchronisation protocol is built around a generation counter and a
binary semaphore. When `execute` is called it:

1. Takes the start mutex.
2. Moves the function object into the shared `fn_` slot.
3. Resets the `finished_` atomic counter to zero.
4. Increments the generation counter.
5. Drops the mutex and calls `start_cv_.notify_all()`.
6. Acquires the binary semaphore, which blocks until it is released.

Each worker runs in a loop that:

1. Waits on the condition variable until the generation changes or stop
   is set.
2. Copies the current generation into its local copy.
3. Calls `fn_(id)` outside the lock.
4. Increments `finished_` with `fetch_add`. If the new value equals
   `n_threads`, it releases the semaphore.

This gives a single mutex acquisition, one condition variable notify,
and one semaphore release per parallel task, which is on the order of a
few microseconds on Linux and is the absolute floor for any OpenMP style
pool. Workers are never spawned on the hot path and never deallocated
until the pool is destroyed.

The pool is neither copyable nor movable (it owns joinable threads).
Destruction sets the stop flag, wakes all workers, and relies on the
`std::jthread` destructors to join them automatically.

## 12. Partition

A `partition<LayoutTag, I, V, BufferTag, BufferN, LP>` in
[partition.hpp](../include/spira/parallel/partition.hpp) is the
thread-local unit of storage in a `parallel_matrix`. It holds:

- `row_start` and `row_end` in the global row index space.
- `std::vector<row_type> rows`: the rows this partition owns.
- `csr_storage<LayoutTag, I, V> csr`: a thread local CSR array.
- A `dirty` bitset and a mode flag mirroring the single threaded matrix.

Partitions are disjoint. No two partitions share rows, and no CSR array
is shared across threads. This is the fundamental property that makes
the parallel algorithms lock free on the critical path.

### NNZ balanced partitioning

`compute_partition_boundaries(row_nnz, n_threads)` produces a
`boundaries` vector of length `n_threads + 1` where
`boundaries[t]` is the first row owned by thread `t`. The algorithm is:

1. Compute the prefix sum of `row_nnz`.
2. For each thread `t`, binary search the prefix sum for the first row
   whose cumulative nnz is at least `(total_nnz * t) / n_threads`.

This gives every thread approximately equal work in terms of non zeros,
not equal row counts. On a matrix where one row has 10,000 non zeros and
all other rows have 10, a row balanced split would leave one thread
doing all the work; the nnz balanced split puts that row alone on one
thread and the rest on the others.

See [06-partition-layout.puml](diagrams/06-partition-layout.puml) for the
resulting memory layout.

## 13. Parallel matrix

`parallel_matrix<LayoutTag, I, V, BufferTag, BufferN, LP, IP, StagingN>` in
[parallel_matrix.hpp](../include/spira/parallel/parallel_matrix.hpp)
is the multi threaded container. It owns the thread pool and the vector
of partitions.

### Insert routing

`insert(r, c, v)` picks the owning partition by integer division:
`owner(r) = r * n_threads / n_rows`. This is a simple uniform split by
row count, which is fine as long as the user expects it; the nnz
balanced split only applies to the partition boundaries stored in the
partitions themselves after construction or rebalance.

If the insert policy is `direct`, the insert writes to
`partitions_[owner].rows[r - row_start].buffer` directly. If it is
`staged`, the insert goes into a per partition staging array first (see
section 15).

### Parallel fill

`parallel_fill(fn)` is an alternative entry point for bulk loads. It
calls `pool_->execute([](tid) { fn(p.rows, p.row_start, p.row_end, tid); })`
on every partition. The user's callback knows the row range and the
partition's row vector and fills them directly. This avoids the
integer division routing entirely and is the fastest way to load a
matrix when the input data is already shardable by row.

### Parallel lock

`lock()` runs the thread pool with a task that calls `partition::lock()`
on each partition. Each partition independently sorts and dedupes its
row buffers and builds or merges its local CSR. No synchronisation is
needed after the partition step because the algorithms that come next
operate on disjoint CSR arrays.

The dirty bitset works the same way as in the single threaded matrix,
just scoped per partition. Only partitions that received inserts since
the last lock do any real merge work; the rest do a memcpy per row.

### Parallel open

`open()` is symmetric. It dispatches `partition::open()` to each
partition, which clears the locked flag and re-enables inserts. Under
`compact_move` the row buffers are re-allocated at this point, which
is the one O(n) cost of re-opening. Under `compact_preserve` the
buffers were never freed so re-opening is O(1).

### Rebalance

`rebalance()` recomputes the partition boundaries based on the current
per row nnz counts, then redistributes rows across partitions. This is
an expensive operation (all rows are moved) and is exposed as an
explicit user action rather than automatic behaviour. It is the right
thing to call when the nnz distribution has drifted far from what it
was at construction.

### Partition access

`partition_at(t)` returns a const reference to partition `t`. Algorithms
iterate `for (size_t t = 0; t < n_threads; ++t) partition_at(t)...` to
get at thread local CSR arrays for read only operations. The matrix
never returns a mutable reference to a partition from outside.

## 14. Parallel algorithms

The parallel algorithms in
[include/spira/parallel/algorithms/](../include/spira/parallel/algorithms/)
mirror the serial set one to one:

```
parallel::algorithms::spmv(mat, x, y);
parallel::algorithms::spgemm(A, B);
parallel::algorithms::transpose(A);
parallel::algorithms::matrix_add(A, B);
parallel::algorithms::accumulate(A, row);
parallel::algorithms::scale(A, s);
```

Each one is a thin wrapper around `pool_->execute` that dispatches the
equivalent serial algorithm to each partition. For SpMV this looks like:

```cpp
pool->execute([&](size_t tid) {
    auto &p = partitions_[tid];
    for (size_t i = 0; i < p.rows.size(); ++i) {
        size_t r = p.row_start + i;
        y[r] = spira::serial::algorithms::sparse_dot_row(p.csr, i, x);
    }
});
```

When the layout, index type, and value type line up for the SIMD path,
`sparse_dot_row` ends up calling `kernel::sparse_dot_float` or
`kernel::sparse_dot_double` directly through the function pointer set at
startup. This is where the parallel and SIMD layers compose: the outer
structure is partition driven, the inner loop is SIMD.

See [04-spmv-sequence.puml](diagrams/04-spmv-sequence.puml) for the full
call sequence from a user `spmv()` call down to the SIMD kernel.

## 15. Insert staging policy

The `insert_policy` template parameter controls how inserts reach the row
buffers in the parallel matrix.

**`direct`** writes to `partitions_[owner].rows[r - row_start].buffer` on
every call. It is zero overhead in the best case, but random row order
produces random partition access, which means the main thread is
constantly jumping between cache lines that belong to different
partitions. On a large matrix this can cost a noticeable fraction of the
insert budget.

**`staged`** inserts first go into a small per partition staging array of
`StagingN` entries. The staging arrays are contiguous and fit in L1 or
L2, so the hot insert loop stays cache resident. Once a staging array
fills up, it is flushed to its partition's row buffers in a single
sequential pass. The flush is also triggered by `lock()` to make sure no
inserts are left in the staging arrays when the matrix is frozen.

The staging array lives on the main thread, not the worker threads. The
worker threads are dormant during insertion; they only come alive on
`lock()` and algorithm calls.

The flush is a simple per partition scatter, not a parallel one, because
the staging array is small and the cost of dispatching a pool task for
it would dominate. A typical `StagingN` is 256 entries, which fits in
roughly four cache lines per partition.

## 16. Putting it together: a full SpMV trace

To tie the layers together, here is what happens end to end when a user
calls `parallel::algorithms::spmv(A, x, y)` on a locked parallel matrix.

1. `spmv` retrieves the thread pool from the matrix.
2. It calls `pool.execute(fn)` where `fn` is a lambda that closes over
   the partitions vector, the input `x`, and the output `y`.
3. `execute` takes the start mutex, moves the lambda into the shared
   slot, resets the finished counter, bumps the generation, drops the
   mutex, and calls `notify_all`.
4. All worker threads wake on the condition variable. Each one observes
   the new generation, copies it, and calls `fn(tid)` outside the lock.
5. Inside `fn`, each thread retrieves its own partition `p`. It iterates
   over the rows of the partition. For each row it retrieves the CSR
   slice (pointer arithmetic into the partition's local CSR array) and
   calls the serial row dot product.
6. If the matrix is SoA with `uint32_t` and `float`, the serial row dot
   product function dispatches to `kernel::sparse_dot_float`, which is
   a function pointer set at program load by the dispatch layer. On a
   machine with AVX2 it points at `sparse_dot_float_avx`, which unrolls
   the loop and uses FMA to accumulate 8 floats per iteration.
7. The kernel writes the result to `y[p.row_start + i]`. Because each
   partition owns a disjoint slice of `y`, there is no contention.
8. Each worker increments `finished_` when it is done with its partition.
   The last worker releases the binary semaphore.
9. `execute` returns control to `spmv`, which returns control to the
   user.

No locks, no atomics, no branches were taken on the inner loop. The only
synchronisation was the initial broadcast in `execute` and the single
semaphore release at the end. The SIMD kernel selection was a single
indirect function call, decided at program load time.

That is the full stack: concepts, traits, layout tags, buffers, rows,
CSR, the matrix, the lock pipeline, the SIMD kernels, the thread pool,
the partitions, the parallel matrix, the parallel algorithms. Every
layer does one thing and hands its result to the next.
