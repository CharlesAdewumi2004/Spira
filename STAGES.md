# Spira — Implementation Stages

Spira is a C++23 header-only dynamic sparse matrix library developed in four incremental stages, each building on the last. This document covers every stage independently: its design goals, data structures, algorithms, and API.

---

## Table of Contents

1. [Stage 1 — MVP](#stage-1--mvp)
2. [Stage 2 — SIMD Kernels & Hardware Detection](#stage-2--simd-kernels--hardware-detection)
3. [Stage 3 — Layout-Aware CSR](#stage-3--layout-aware-csr)
4. [Stage 4 — Multi-Threading](#stage-4--multi-threading)
5. [Progression Summary](#progression-summary)

---

## Stage 1 — MVP

**Branch:** `stage1/MVP`

### Goal

Establish the core abstraction: a dynamic sparse matrix that supports efficient random insertion and sequential read without requiring up-front structure.

### Core Design

Each row maintains two storage layers — a **write buffer** for fast unsorted inserts and a **sorted slab** for fast reads. Flushing the buffer into the slab is triggered lazily on the first read (or explicitly). The matrix exposes three operational **modes** that trade buffer capacity for read latency.

```
┌─────────────────────────────────────────────────────────────────┐
│                        matrix<L, I, V>                          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐  │
│  │  row[0]  │  │  row[1]  │  │  row[2]  │  ...  │  row[n]  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘  │
│       │              │              │                   │        │
│  ┌────▼─────────────────────────────────────────────────▼────┐  │
│  │  Each row owns:                                           │  │
│  │                                                           │  │
│  │   ┌────────────────┐     ┌─────────────────────────┐    │  │
│  │   │  Write Buffer  │     │     Sorted Slab          │    │  │
│  │   │  (unsorted)    │────▶│  (col, val) ascending   │    │  │
│  │   │  insert_heavy: │flush│  Binary search reads    │    │  │
│  │   │  hash map 16K  │     │  Last-write-wins dedup  │    │  │
│  │   │  balanced: 128 │     │                         │    │  │
│  │   │  spmv:      32 │     │                         │    │  │
│  │   └────────────────┘     └─────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Matrix Modes

The mode controls buffer capacity and the threshold at which a flush occurs:

| Mode           | Buffer Capacity | Designed For              |
|----------------|-----------------|---------------------------|
| `spmv`         | 32 entries      | Read-heavy (SpMV loops)   |
| `balanced`     | 128 entries     | Mixed insert + read       |
| `insert_heavy` | 16 384 (hash)   | Bulk assembly passes      |

```cpp
// Switch modes at any point
mat.set_mode(spira::config::matrix_mode::insert_heavy); // bulk fill
// ... millions of inserts ...
mat.set_mode(spira::config::matrix_mode::spmv);         // compute phase
```

### Layout Tags

All stages share two compile-time layout strategies selected via a template parameter:

```
  AoS (Array of Structs)          SoA (Struct of Arrays)
  ┌───┬───┬───┬───┬───┐          ┌───┬───┬───┬───┬───┐
  │c₀ │v₀ │c₁ │v₁ │c₂ │          │c₀ │c₁ │c₂ │c₃ │c₄ │  cols[]
  └───┴───┴───┴───┴───┘          └───┴───┴───┴───┴───┘
  Interleaved — cache-friendly    ┌───┬───┬───┬───┬───┐
  for mixed col+val access        │v₀ │v₁ │v₂ │v₃ │v₄ │  vals[]
                                  └───┴───┴───┴───┴───┘
                                  Separate — SIMD-friendly
```

### Type Constraints

```cpp
// I must be an unsigned integer ≥ 32 bits (uint32_t, uint64_t, ...)
template <typename I>
concept Indexable = std::unsigned_integral<I> && sizeof(I) >= 4;

// V must support arithmetic and have a zero() sentinel
template <typename V>
concept Valueable = /* +, *, +=, *=, ValueTraits<V>::zero(), is_zero() */;
```

### Template Signature

```cpp
template <
    class LayoutTag,             // layout::tags::aos_tag or soa_tag
    concepts::Indexable I = uint32_t,
    concepts::Valueable V = double
>
class matrix;
```

### Key API (Stage 1)

```cpp
spira::matrix<spira::layout::tags::soa_tag, uint32_t, double> A(1000, 1000);

// Insertion (buffered, O(1) amortised)
A.insert(row, col, value);

// Read (flushes buffer on first call if needed)
double v     = A.get(row, col);        // 0.0 if absent
bool   found = A.contains(row, col);
size_t nnz   = A.row_nnz(row);

// Mode switch
A.set_mode(spira::config::matrix_mode::spmv);

// Arithmetic operators
auto C = A + B;         // matrix addition
auto C = A * B;         // SpGEMM
auto y = A * x;         // SpMV (returns std::vector)
auto T = ~A;            // transpose
auto S = A * 2.5;       // scalar multiply
```

### Algorithms (Stage 1)

| Algorithm   | Signature                                          |
|-------------|---------------------------------------------------|
| SpMV        | `spmv(A, x, y)` — `y = A × x`                   |
| SpGEMM      | `spgemm(A, B)` — returns `C = A × B`             |
| Transpose   | `transpose(A)` — returns `Aᵀ`                    |
| Addition    | `matrix_add(A, B)` — returns `A + B`             |
| Scale       | `scale(A, s)` — returns `s × A`                  |

---

## Stage 2 — SIMD Kernels & Hardware Detection

**Branch:** `stage3/simd-kernels-and-prefetch`

### Goal

Replace the scalar sparse dot-product inner loop with architecture-specific SIMD kernels selected at runtime. Add CPU feature detection and adaptive memory-latency measurement so the optimal kernel and prefetch distance are chosen automatically on any machine.

### Architecture

A **dispatch layer** runs once at program startup (via a `static` initialiser). It probes CPU features, measures DRAM latency, and installs function pointers that the hot path calls directly — zero branches on the critical path.

```
Program start
      │
      ▼
┌─────────────────────────────────────────────┐
│              dispatch.cpp (once)             │
│                                              │
│  1. detect_cpu_features()                   │
│     ├─ x86: CPUID leaves 1, 7 + XSAVE check │
│     └─ ARM: hwcap / sysctl / WinAPI          │
│                                              │
│  2. measure_memory_latency()                 │
│     └─ pointer-chase chain → DRAM ns / cyc  │
│                                              │
│  3. select best kernel                       │
│     AVX-512 > AVX2+FMA > SSE4.2 > NEON      │
│                   │                          │
│  4. write to function pointers               │
│     kernel::sparse_dot_double = &dot_avx2   │
│     kernel::sparse_dot_float  = &dot_avx2f  │
└─────────────────────────────────────────────┘
                    │
                    ▼  (hot path — no branching)
          y[i] = sparse_dot_float(vals, cols, x, n)
```

### Hardware Detection

```
 CPU
  │
  ├─── x86 / x64 ────────────────────────────────────────────────────┐
  │    CPUID leaf 1:   SSE2, SSE4.2, AVX, FMA                        │
  │    CPUID leaf 7:   AVX2, AVX-512F/BW/VL/DQ                       │
  │    XSAVE:          OS saves YMM/ZMM registers (required for AVX)  │
  │                                                                    │
  ├─── ARM64 ─────────────────────────────────────────────────────────┤
  │    Linux:          /proc/cpuinfo / getauxval(AT_HWCAP)            │
  │    Apple:          sysctlbyname("hw.optional.AdvSIMD")            │
  │    Windows:        IsProcessorFeaturePresent()                     │
  │                                                                    │
  └─── All ────────────────────────────────────────────────────────────┘
       Cache sizes: L1d/L1i/L2/L3 — used for prefetch distance tuning
```

### Adaptive Prefetching

DRAM latency is measured with a **pointer-chase chain** (linked list of random pointers spanning memory, forces cache misses):

```
prefetch_distance = ceil( dram_latency_cycles × stride_bytes
                          ─────────────────────────────────── )
                           bytes_per_iter × cycles_per_iter
```

This gives the number of elements to prefetch ahead so a `__builtin_prefetch` request arrives in DRAM just before the element is needed.

### Kernel Hierarchy

```
                    ┌─ CPU supports AVX-512F + OS saves ZMM?
                    │         YES → dot_avx512  (8×float / 4×double per iter)
                    │
                    ├─ CPU supports AVX2 + FMA?
                    │         YES → dot_avx2    (8×float / 4×double per iter, FMA fused)
                    │
                    ├─ CPU supports SSE4.2?
                    │         YES → dot_sse     (4×float / 2×double per iter)
                    │
                    ├─ CPU has NEON (ARM)?
                    │         YES → dot_neon    (4×float / 2×double per iter)
                    │
                    └─ Fallback
                              → dot_scalar (1 element per iteration)
```

Each kernel uses a **scalar gather** (manual index load) because sparse column indices are irregular — this is the correct pattern for sparse BLAS:

```
// Conceptually (AVX2 path for float):
for each block of 8 non-zeros:
    __m256  v = load8(vals + k);          // sequential values load
    __m256  x_gathered = {x[cols[k+0]], x[cols[k+1]], ..., x[cols[k+7]]};
    acc = _mm256_fmadd_ps(v, x_gathered, acc);
```

### Function Pointers

```cpp
namespace spira::kernel {
    // Set by dispatch() at startup, called on every SpMV row
    extern double (*sparse_dot_double)(
        const double  *vals, const uint32_t *cols,
        const double  *x,    size_t n, size_t x_size);

    extern float  (*sparse_dot_float)(
        const float   *vals, const uint32_t *cols,
        const float   *x,    size_t n, size_t x_size);
}
```

### SpMV Integration

For `soa_tag + uint32_t + float/double` the compiler selects a specialisation that calls the kernel directly via the function pointer. All other type/layout combinations fall back to the generic element-by-element loop.

```cpp
// Generic path (all layouts, all types)
template <class L, class I, class V>
void spmv(const matrix<L,I,V> &A, const vector<V> &x, vector<V> &y);

// SIMD path (SoA + uint32_t + float)  — zero overhead vs generic
template <>
void spmv<soa_tag, uint32_t, float>(...) {
    A.for_each_row([&](const auto &row, uint32_t i) {
        y[i] = kernel::sparse_dot_float(
                   row.val_data(), row.col_data(), x.data(), row.size(), N);
    });
}
```

### New Files (Stage 2)

```
src/kernels/
├── dispatch.cpp                 startup: detect → measure → install
├── hw_detect.hpp                CpuFeatures + MemoryLatencyInfo structs
├── dot_impls/
│   ├── dot_scalar.cpp           scalar fallback
│   ├── x86/
│   │   ├── dot_sse.cpp          SSE4.2 kernel
│   │   ├── dot_avx.cpp          AVX2+FMA kernel
│   │   └── dot_avx512.cpp       AVX-512 kernel
│   └── arm/
│       └── dot_neon.cpp         NEON kernel
```

---

## Stage 3 — Layout-Aware CSR

**Branch:** `stage3/layout-aware-csr`

### Goal

Replace the mode-switching buffer design with a formal **open / locked** lifecycle. In open mode the matrix accepts inserts into per-row staging buffers; calling `lock()` sorts, deduplicates, and merges all buffers into a flat **Compressed Sparse Row (CSR)** array, giving zero-overhead reads with contiguous memory access.

### Open / Locked Lifecycle

```
                    ┌─────────────────────────────────┐
                    │           open mode              │
                    │  insert(row, col, val)           │
                    │  Per-row buffer: unsorted,       │
                    │  last-write-wins on same column  │
                    └──────────────┬──────────────────┘
                                   │  lock()
                                   ▼
             ┌────────────────────────────────────────────┐
             │                lock()                       │
             │                                             │
             │  ① sort + dedup + filter each row buffer   │
             │     O(k log k)  per row                    │
             │                                             │
             │  ② build_csr  (first lock)                 │
             │     or merge_csr  (subsequent locks)        │
             │     O(nnz_old + nnz_new)  two-pointer merge│
             │                                             │
             │  ③ install per-row CSR slices (pointers)   │
             │     O(n_rows)                               │
             │                                             │
             │  ④ clear staging buffers                   │
             └──────────────┬─────────────────────────────┘
                            │
                            ▼
                    ┌─────────────────────────────────┐
                    │          locked mode             │
                    │  get / contains / spmv           │
                    │  Direct pointer into flat CSR    │
                    │  Binary search per-row slice     │
                    │  Zero allocation on reads        │
                    └──────────────┬──────────────────┘
                                   │  open()
                                   ▼
                              open mode again
                    (CSR preserved as committed history)
```

### CSR Storage Layout

The flat CSR is a single contiguous allocation shared by all rows, with a `[n_rows+1]` offsets array as the row index.

**SoA CSR** (`layout::tags::soa_tag`):

```
offsets:  [ 0,  2,  5,  5,  8 ]     ← n_rows+1 entries
           row0 row1 row2 row3

cols:     [ 3, 7, 1, 4, 9, 2, 6, 8 ]   ← nnz entries, contiguous
vals:     [ …, …, …, …, …, …, …, … ]   ← nnz entries, contiguous

row 1 lives at cols[2..4], vals[2..4]   ← pointer arithmetic, no copy
```

**AoS CSR** (`layout::tags::aos_tag`):

```
offsets:  [ 0,  2,  5,  5,  8 ]

pairs:    [{3,v},{7,v},{1,v},{4,v},{9,v},{2,v},{6,v},{8,v}]
           └──row 0───┘└────── row 1 ─────┘     └── row 3 ──┘
           col+val interleaved — one cache line carries both
```

Both layouts are **64-byte aligned** and allocated with `std::aligned_alloc`.

### Merge Strategy

On `lock()` after the first cycle, new buffer entries must be merged with existing CSR data.

```
Old CSR row i:   [ 1  4  7  9 ]      (sorted, from prior lock)
New buffer row i:[ 3  4  8  ]        (sorted by row::lock())

Two-pointer merge (buffer wins on collision):

  old:  1  4  7  9
  new:     4     8
            ↓
  out:  1  4  7  8  9

Allocation strategy:
  total_ub = old.nnz + Σ buf_nnz[i]

  total_ub ≤ old.capacity  →  reuse path (reverse-order in-place merge)
  total_ub > old.capacity  →  grow path  (1.5× new allocation, forward merge)
```

### Dirty Bitset (Dynamic Optimisation)

A `std::vector<bool> dirty_` tracks which rows have pending buffer changes since the last `lock()`. On the next `lock()`, clean rows skip the per-entry merge loop and use a single `memmove` / `memcpy` instead — critical when only a small fraction of rows change between lock cycles.

```
dirty_:  [ 0, 1, 0, 0, 1, 0, 0, 1 ]
                 ↓           ↓       ↓
           bulk copy    merge    merge
           (fast)      (full)   (full)
```

### O(1) Buffer Lookups

An `ankerl::unordered_dense::map<I, size_t>` index inside each buffer always points to the **last-written entry** for each column, so `get`, `contains`, and `accumulate` are O(1) / O(unique cols) in open mode rather than O(k) backward scan.

```
Buffer state after inserts (insertion order):
  [(col=5, v=1.0), (col=2, v=3.0), (col=5, v=2.0)]

index_:  { 2 → 1,   5 → 2 }       ← always points to last write

get(col=5) → buf_[index_[5]].value == 2.0    O(1)
accumulate → iterate index_, sum buf_[idx].value
```

### New Template Parameters (Stage 3)

```cpp
template <
    class LayoutTag,
    concepts::Indexable   I       = uint32_t,
    concepts::Valueable   V       = double,
    class BufferTag               = buffer::tags::array_buffer<layout::tags::aos_tag>,
    std::size_t           BufferN = 64,           // initial buffer reserve hint
    config::lock_policy   LP      = config::lock_policy::compact_preserve
>
class matrix;
```

**Lock policies:**

| Policy             | After `lock()`                              | After `open()`        |
|--------------------|--------------------------------------------|-----------------------|
| `compact_preserve` | CSR built; per-row buffers kept (zero-size)| O(1) — just flip flag |
| `compact_move`     | CSR built; per-row buffers freed           | O(n) — re-allocate    |
| `no_compact`       | Sorted buffer used directly (no CSR)       | O(1)                  |

### Key API (Stage 3 additions)

```cpp
spira::matrix<spira::layout::tags::soa_tag> M(1000, 1000);

// --- Open mode ---
M.insert(0, 5, 1.5f);
M.insert(0, 5, 2.0f);   // overwrites col 5 (last-write-wins)

// --- Lock ---
M.lock();                // sort → dedup → build CSR

// --- Locked mode (zero-allocation reads) ---
float v = M.get(0, 5);  // 2.0f
bool  b = M.contains(0, 5);
const auto *csr = M.csr();  // raw pointer to CSR storage

// --- Re-open ---
M.open();                // CSR preserved; buffer ready for new inserts
M.insert(0, 7, 3.0f);
M.lock();                // merge: old CSR + new buffer entry
```

### New Files (Stage 3)

```
include/spira/matrix/
├── storage/
│   ├── csr_storage.hpp          csr_storage<L,I,V> + csr_slice<L,I,V>
│   └── csr_build.hpp            build_csr() + merge_csr() templates
├── buffer/
│   ├── aos_array_buffer.hpp     AoS buffer + index_ map
│   └── soa_array_buffer.hpp     SoA buffer + index_ map
├── matrix.hpp                   updated template + lock/open + dirty_
└── row.hpp                      updated: csr_slice install/reset
```

---

## Stage 4 — Multi-Threading

**Branch:** `stage4/MutilThreaded`

### Goal

Scale to multi-core machines by statically partitioning rows across a fixed-size thread pool. Each partition owns its own CSR and row objects, so lock/unlock and SpMV run with no synchronisation on the critical path.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   parallel_matrix<L,I,V,...>                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ partition[0] │  │ partition[1] │  │ partition[2] │  ...     │
│  │ rows 0..249  │  │ rows 250..499│  │ rows 500..749│          │
│  │              │  │              │  │              │          │
│  │  row[0]      │  │  row[250]    │  │  row[500]    │          │
│  │  row[1]      │  │  row[251]    │  │  row[501]    │          │
│  │  ...         │  │  ...         │  │  ...         │          │
│  │  csr (local) │  │  csr (local) │  │  csr (local) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│  ┌──────▼──────────────────▼──────────────────▼───────────────┐ │
│  │                     thread_pool                             │ │
│  │   worker[0]           worker[1]           worker[2]        │ │
│  │   (owns partition[0]) (owns partition[1]) (owns partition[2])│ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Thread Pool Design

Workers are created once at construction and sleep on a condition variable between `execute()` calls. No threads are ever spawned on the hot path.

```
Main thread calls execute(fn):
  ① increment generation counter
  ② notify_all on start_cv
  ③ block on binary semaphore (done_)

Each worker:
  ① wakes when generation changes
  ② calls fn(thread_id)
  ③ decrements finished counter
  ④ last worker releases done_ semaphore

Total synchronisation overhead: 1 mutex lock + 1 cond notify + 1 semaphore
```

### NNZ-Balanced Partitioning

Rows are assigned to threads so each thread handles approximately the same number of non-zeros (not the same number of rows). This prevents load imbalance when row densities vary widely.

```
Row NNZ:    [ 2, 8, 1, 5, 3, 9, 2, 4, 6, 1 ]
             ─────────────────────────────────
Total NNZ:  41     (target per thread: 41/3 ≈ 14)

Thread 0:   rows 0-2  → NNZ  2+8+1 = 11  (close to 14)
Thread 1:   rows 3-5  → NNZ  5+3+9 = 17
Thread 2:   rows 6-9  → NNZ  2+4+6+1 = 13

Computed via prefix-sum + binary search:  O(n_rows + n_threads·log(n_rows))
```

### Parallel Lock Cycle

```
parallel_matrix::lock():
  pool.execute([](partition &p, size_t tid) {
      for (auto &r : p.rows)
          r.lock();                     // per-row sort+dedup
      if (p.csr.is_built())
          p.csr = merge_csr(p.rows, move(p.csr), p.dirty);
      else
          p.csr = build_csr(p.rows);    // first lock
      // install CSR slices + clear buffers
  });
  // No barrier needed — partitions are disjoint
```

### Insert Policies

Stage 4 adds an optional staging layer to improve cache behaviour on bulk inserts.

```
insert_policy::direct  (default):
  mat.insert(row=5, col=3, val=1.0)
  → immediately written to partition[owner_of_row_5].rows[5].buffer
  → simple, zero overhead, but random partition access = cache misses

insert_policy::staged  (optional):
  per-thread staging array [StagingN entries] in L1/L2
  flush automatically when full or on lock()
  → hot insertion loop stays cache-resident
  → useful for scatter-heavy fill patterns
```

### New Template Parameters (Stage 4)

```cpp
template <
    class LayoutTag,
    concepts::Indexable     I        = uint32_t,
    concepts::Valueable     V        = double,
    class BufferTag                  = buffer::tags::array_buffer<layout::tags::aos_tag>,
    std::size_t             BufferN  = 64,
    config::lock_policy     LP       = config::lock_policy::compact_preserve,
    config::insert_policy   IP       = config::insert_policy::direct,
    std::size_t             StagingN = 256      // entries per staging buffer
>
class parallel_matrix;
```

### Key API (Stage 4)

```cpp
using PM = spira::parallel::parallel_matrix<
    spira::layout::tags::soa_tag,
    uint32_t, double,
    spira::buffer::tags::array_buffer<spira::layout::tags::aos_tag>,
    64,
    spira::config::lock_policy::compact_preserve,
    spira::config::insert_policy::direct
>;

PM mat(10'000, 10'000, /*n_threads=*/8);

// Single-threaded insert (routes to owning partition)
for (auto [r, c, v] : triples)
    mat.insert(r, c, v);

// Or parallel fill (no routing overhead, user controls locality)
mat.parallel_fill([&](auto &rows, size_t r_start, size_t r_end, size_t tid) {
    for (auto [r, c, v] : my_shard[tid])
        rows[r - r_start].insert(c, v);
});

mat.lock();   // parallel: each thread locks its own partition

std::vector<double> x(10'000, 1.0), y(10'000);
spira::parallel::algorithms::spmv(mat, x, y);  // parallel SpMV

mat.open();
mat.insert(42, 7, 9.9);   // selective update
mat.lock();               // only partition owning row 42 does real merge work
                          // (dirty bitset skips all other rows)

mat.rebalance();          // redistribute rows if NNZ has drifted
```

### Parallel Algorithms

Both `serial/` and `parallel/` namespaces expose identical signatures:

| Algorithm        | Serial                        | Parallel                           |
|------------------|-------------------------------|------------------------------------|
| SpMV             | `serial::spmv(mat, x, y)`    | `parallel::spmv(mat, x, y)`       |
| SpGEMM           | `serial::spgemm(A, B)`       | `parallel::spgemm(A, B)`          |
| Transpose        | `serial::transpose(A)`       | `parallel::transpose(A)`          |
| Matrix addition  | `serial::matrix_add(A, B)`   | `parallel::matrix_add(A, B)`      |
| Accumulate       | `serial::accumulate(A, row)` | `parallel::accumulate(A, row)`    |
| Scale            | `serial::scale(A, s)`        | `parallel::scale(A, s)`           |

### SIMD Integration

The SIMD kernel function pointers from Stage 2 are called inside the parallel SpMV for `soa_tag + uint32_t + float/double` specialisations — the combination of partitioned parallelism and SIMD provides the maximum throughput path:

```
Parallel SpMV (SoA + float):

  thread 0 → rows 0..249   → sparse_dot_float (AVX2 kernel) per row
  thread 1 → rows 250..499 → sparse_dot_float (AVX2 kernel) per row
  ...                               ↑
                               function pointer set at startup by dispatch()
```

### New Files (Stage 4)

```
include/spira/
├── parallel/
│   ├── thread_pool.hpp          fixed worker pool, generation counter
│   ├── partition.hpp            row range + local CSR + rebalance
│   ├── parallel_matrix.hpp      main parallel container
│   ├── insert_staging.hpp       optional per-thread staging buffers
│   ├── parallel.hpp             execute() utility
│   └── algorithms/
│       ├── spmv.hpp             parallel SpMV + SIMD specialisation
│       ├── spgemm.hpp           parallel SpGEMM
│       ├── matrix_addition.hpp  parallel A + B
│       ├── transpose.hpp        parallel transpose
│       ├── accumulate.hpp       parallel row accumulate
│       └── scalars.hpp          parallel scalar multiply/divide
└── serial/
    ├── serial.hpp               umbrella include
    ├── spmv.hpp
    ├── spgemm.hpp
    ├── matrix_addition.hpp
    ├── transpose.hpp
    ├── accumulate.hpp
    └── scalars.hpp
```

---

## Progression Summary

```
Stage 1: MVP
  Dynamic sparse matrix — per-row buffer + sorted slab
  Mode-based buffer sizing (spmv / balanced / insert_heavy)
  Sequential algorithms: SpMV, SpGEMM, Transpose, Add, Scale
           │
           │  +SIMD kernels
           │  +CPU feature detection (CPUID / hwcap / sysctl)
           │  +Adaptive prefetch distance (DRAM latency measurement)
           ▼
Stage 2: SIMD Kernels & Hardware Detection
  Runtime kernel dispatch: AVX-512 > AVX2+FMA > SSE4.2 > NEON > scalar
  Zero hot-path branches (function pointers set at startup)
  SpMV specialisation for soa_tag + uint32_t + float/double
           │
           │  +open / locked lifecycle (replaces 3-mode system)
           │  +flat CSR with 64-byte aligned SoA or AoS arrays
           │  +two-pointer merge on re-lock (1.5× growth)
           │  +dirty bitset (skip unchanged rows on lock)
           │  +O(1) buffer get / contains (index map)
           ▼
Stage 3: Layout-Aware CSR
  Formal open → lock → open cycle
  Single contiguous CSR allocation shared across all rows
  Lock policy: compact_preserve / compact_move / no_compact
  Per-row dirty tracking; bulk memmove for clean rows
           │
           │  +static row partitioning across n_threads
           │  +thread-local CSR per partition (no shared state)
           │  +NNZ-balanced partition boundaries
           │  +thread pool (sleep/wake, zero allocation on hot path)
           │  +insert_policy::staged for cache-resident bulk fills
           │  +parallel_fill() for lock-free parallel assembly
           │  +rebalance() after workload shift
           ▼
Stage 4: Multi-Threading
  parallel_matrix<L,I,V,..., n_threads>
  Parallel lock, parallel SpMV, parallel SpGEMM
  SIMD kernels inside parallel dispatch
  Serial and parallel algorithm namespaces with identical signatures
```

### Feature Comparison

| Feature                  | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|--------------------------|:-------:|:-------:|:-------:|:-------:|
| Dynamic insert           | ✓       | ✓       | ✓       | ✓       |
| AoS / SoA layouts        | ✓       | ✓       | ✓       | ✓       |
| SpMV / SpGEMM / Transpose| ✓       | ✓       | ✓       | ✓       |
| SIMD kernels             |         | ✓       | ✓       | ✓       |
| CPU feature detection    |         | ✓       | ✓       | ✓       |
| Adaptive prefetch        |         | ✓       | ✓       | ✓       |
| Open / locked lifecycle  |         |         | ✓       | ✓       |
| Flat CSR (contiguous)    |         |         | ✓       | ✓       |
| Two-pointer CSR merge    |         |         | ✓       | ✓       |
| Dirty bitset             |         |         | ✓       | ✓       |
| O(1) buffer lookup       |         |         | ✓       | ✓       |
| Thread pool              |         |         |         | ✓       |
| NNZ-balanced partitions  |         |         |         | ✓       |
| Thread-local CSR         |         |         |         | ✓       |
| Parallel lock / SpMV     |         |         |         | ✓       |
| Staged insert policy     |         |         |         | ✓       |
| Parallel fill            |         |         |         | ✓       |
| `rebalance()`            |         |         |         | ✓       |
