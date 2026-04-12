# Spira — Benchmark Suite

This document describes the performance benchmarks for each Spira stage in detail: what is measured, how the measurement is structured, what the reported metrics mean, and how to interpret results. All benchmarks use [Google Benchmark](https://github.com/google/benchmark) and are built and run via `bench/run_benchmark.sh`.

---

## Table of Contents

1. [Benchmark Infrastructure](#benchmark-infrastructure)
   - [Cold-Cache Methodology](#cold-cache-methodology)
   - [Matrix Configuration](#matrix-configuration)
   - [Access Patterns](#access-patterns)
   - [Reported Metrics](#reported-metrics)
2. [Stage 1 — Insert & SpMV (MVP)](#stage-1--insert--spmv-mvp)
3. [Stage 2 — SpMV with SIMD Dispatch](#stage-2--spmv-with-simd-dispatch)
4. [Stage 3 — Insert & SpMV (Layout-Aware CSR)](#stage-3--insert--spmv-layout-aware-csr)
5. [Stage 4 — Parallel Insert, SpMV & Thread Scaling](#stage-4--parallel-insert-spmv--thread-scaling)
6. [Runner Script](#runner-script)
7. [Interpreting Results](#interpreting-results)

---

## Benchmark Infrastructure

### Cold-Cache Methodology

Every benchmark times individual iterations of a tight loop. Inside the loop, `state.PauseTiming()` is called before `flush_cache()` and `state.ResumeTiming()` is called immediately before the operation under test. This ensures each iteration starts with a cold LLC (Last Level Cache).

```
Iteration N:
  ┌──────────────────────────────────────────────────┐
  │  PauseTiming()                                   │
  │    flush_cache()  ← stride through LLC-sized     │
  │                     arena, evicting matrix data  │
  │  ResumeTiming()                                  │
  │    ← timed section begins here                   │
  │    operation(...)  ← pays full DRAM cost          │
  │    ← timed section ends here                     │
  └──────────────────────────────────────────────────┘
```

**Why cold-cache measurements?**  
SpMV is memory-bound: arithmetic intensity ≈ 2 FLOP / 12 B ≈ 0.17 FLOP/B. At 10 % density the matrix alone is ~120 MB, far exceeding any LLC. Cold-cache results reflect the real access cost a solver would pay on first use each timestep, without artificial warm-cache speedup from loop repetition.

**How `flush_cache` works:**

```
flush_cache():
  1. Detect LLC size at first call (static, computed once):
       Windows → GetLogicalProcessorInformation(), scan Level-3 entries
       POSIX   → sysconf(_SC_LEVEL3_CACHE_SIZE), fallback 32 MiB

  2. Allocate static arena of that size (also done once)

  3. Each call: stride through arena in 64-byte steps (one cache line),
     XOR each line into a volatile sink.
     volatile prevents the compiler from eliding the load.
     64-byte stride ensures every cache line is touched exactly once.
```

### Matrix Configuration

All stages benchmark the same logical problem size:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `N` | 10 000 | Rows = Cols. Working sets range from 1 MB (0.1 % density) to 120 MB (10 % density), spanning L3-resident to DRAM-bound. |
| `BATCH` | 256 | Entries per incremental Insert iteration. Models a realistic steady-state update where a small fraction of entries change per solve. |
| `SEED` | 42 | Fixed RNG seed for reproducibility. Full-matrix and batch triples use different seeds (`SEED` vs `SEED ^ 0xDEADBEEF`) to avoid systematic column-index collisions. |

**Density levels:**

| `nnz_per_row` | Fill fraction | Total NNZ | Approx. working set |
|--------------|---------------|-----------|---------------------|
| 10 | 0.1 % | 100 000 | ~1.2 MB |
| 100 | 1 % | 1 000 000 | ~12 MB |
| 1 000 | 10 % | 10 000 000 | ~120 MB |

At 0.1 % the matrix fits in a typical 8–16 MB LLC (warm-ish after one pass). At 10 % it does not fit in any current server LLC, making it a purely DRAM-bound case.

### Access Patterns

Each benchmark is parameterised by `range(1)`:

```
range(1) = 0 → random pattern
  Column indices drawn i.i.d. uniform from [0, N).
  Simulates: unstructured sparsity (graph problems, random connectivity).
  Memory behaviour: irregular gather of x[col]; poor TLB and prefetcher
                    behaviour; stresses DRAM bandwidth.

range(1) = 1 → strided pattern
  Column k of row r = (k × stride) % N   where stride = N / nnz_per_row.
  Simulates: banded / stencil PDE matrices; structured sparsity.
  Memory behaviour: sequential or near-sequential gather; prefetcher-
                    friendly; often shows higher effective bandwidth.
```

The performance gap between random and strided patterns quantifies the TLB/prefetch sensitivity of the implementation and the hardware.

### Reported Metrics

Google Benchmark outputs two throughput columns when both `SetItemsProcessed` and `SetBytesProcessed` are set:

| Column | Unit | What it measures |
|--------|------|-----------------|
| `items_per_second` | ops/s or FLOP/s | **Insert**: insertions/s. **SpMV**: floating-point operations per second (2 FLOP per non-zero: one multiply + one accumulate). |
| `bytes_per_second` | B/s (reported as GB/s) | **Effective memory bandwidth** — bytes that must transit the memory hierarchy per operation, computed from the analytical access model below. |

**SpMV bandwidth model:**

```
bytes per SpMV pass =
    nnz × sizeof(double)        ← CSR value array (read)
  + nnz × sizeof(uint32_t)      ← CSR column-index array (read)
  + (N+1) × sizeof(size_t)      ← CSR row-offset array (read)
  + N × sizeof(double)          ← input vector x (read)
  + N × sizeof(double)          ← output vector y (written)

For N=10 000, nnz_per_row=100 (1 % density):
  = 1 000 000 × 8  +  1 000 000 × 4  +  10 001 × 8  +  10 000 × 8  +  10 000 × 8
  = 8 000 000  +  4 000 000  +  80 008  +  80 000  +  80 000
  ≈ 12.24 MB per pass
```

Comparing `bytes_per_second` against the machine's published STREAM Triad bandwidth gives a **roofline utilisation** figure — how close the implementation is to the hardware memory limit.

**Insert bandwidth model:**

```
bytes per Insert iteration =
    BATCH × (sizeof(uint32_t) + sizeof(double))   ← buffer writes (col + val)
  = 256 × 12
  = 3 072 B  ≈ 3 KB per batch

Note: this is a lower bound. The full merge cost (reading/writing CSR slabs)
is not modelled here; it shows up as increased wall-clock time per item.
```

---

## Stage 1 — Insert & SpMV (MVP)

**Branch:** `stage1/MVP`  
**File:** `bench/spira_bench.cpp`

Stage 1 uses the mode-switching buffer design: `set_mode(insert_heavy)` enables a 16 384-entry hash-map buffer; `flush()` merges it into a sorted slab; `set_mode(spmv)` shrinks the buffer to 32 entries to give the SpMV path zero-overhead row reads.

### InsertFixture

**What is being measured:**  
The cost of inserting BATCH=256 entries into a matrix that is already at target density (steady-state), then calling `flush()` to merge them.

```
SetUp (outside timed loop):
  ① generate N × nnz_per_row full triples
  ② insert all into fresh matrix
  ③ flush() → sorted slab at target density
  ④ set_mode(insert_heavy)   ← ready for timed loop

Timed loop (each iteration):
  PauseTiming → flush_cache → ResumeTiming
  ┌────────────────────────────────────┐
  │  for each t in batch (256 items): │
  │      mat.insert(t.row,t.col,t.val)│  ← O(1) hash-map write
  │  mat.flush()                      │  ← sort + dedup + merge slab
  └────────────────────────────────────┘
```

**Why steady-state?**  
The first `flush()` on an empty matrix only calls `build_csr` (linear scan). All subsequent flushes call a two-pointer merge against existing sorted data — this is the path a real solver takes. Benchmarking only the steady-state merge cost gives a representative throughput figure.

**What `flush()` does internally:**

```
flush():
  1. Sort buffer by column index         O(k log k)
  2. Deduplicate (last-write-wins)       O(k)
  3. Merge with existing slab            O(nnz_old + k)
     ┌─ old slab: [ 1, 4, 7, 9 ]
     ├─ new buf:  [ 3, 4, 8   ]
     └─ result:  [ 1, 3, 4, 7, 8, 9 ]  (buffer wins on col=4)
  4. Install new slab pointer
  5. Clear buffer
```

### SpMVFixture (Stage 1)

**What is being measured:**  
`y = A × x` using the scalar dot-product inner loop (no SIMD in Stage 1).

```
SetUp:
  ① fill and flush matrix to target density
  ② set_mode(spmv)   ← 32-entry buffer, minimises overhead on reads
  ③ generate random x ∈ [0,1)^N, zero y

Timed loop:
  PauseTiming → flush_cache → ResumeTiming
  ┌────────────────────────────────────────┐
  │  spira::algorithms::spmv(mat, x, y)  │
  │                                       │
  │  for each row i:                      │
  │      y[i] = Σ mat[i][j] × x[j]       │  ← scalar dot product
  └────────────────────────────────────────┘
  DoNotOptimize(y.data()) + ClobberMemory()
```

`DoNotOptimize` / `ClobberMemory` prevent the compiler from eliding the computation (the result must be treated as observable).

---

## Stage 2 — SpMV with SIMD Dispatch

**Branch:** `stage2/SIMD-Kernels`  
**File:** `bench/spira_bench.cpp`

Stage 2 retains the Stage 1 insert design and focuses exclusively on SpMV throughput. The only change from Stage 1's SpMV fixture is the kernel called: Stage 2's `spira::algorithms::spmv` routes through a runtime-selected SIMD function pointer installed at program startup by `dispatch()`.

### SpMVFixture (Stage 2)

The fixture setup and timing structure are identical to Stage 1. The difference is internal:

```
Stage 1 SpMV inner loop (scalar):
  for (size_t k = 0; k < nnz; ++k)
      acc += vals[k] * x[cols[k]];    ← 1 element/iter

Stage 2 SpMV inner loop (AVX2, 4×double/iter):
  for (size_t k = 0; k + 4 <= nnz; k += 4) {
      __m256d v = _mm256_loadu_pd(vals + k);
      __m256d g = {x[cols[k]], x[cols[k+1]], x[cols[k+2]], x[cols[k+3]]};
      acc = _mm256_fmadd_pd(v, g, acc);    ← 4 FMA/iter, fused multiply-add
  }
  // scalar tail for remaining elements
```

**Kernel selection at startup:**

```
dispatch() (runs once via static initialiser):
  ├─ AVX-512F available (and OS saves ZMM)?  → dot_avx512  (4×double/iter)
  ├─ AVX2 + FMA available?                   → dot_avx2    (4×double/iter, FMA)
  ├─ SSE4.2 available?                       → dot_sse     (2×double/iter)
  ├─ NEON available (ARM)?                   → dot_neon    (2×double/iter)
  └─ fallback                                → dot_scalar  (1×double/iter)
```

**Why the Stage 2 benchmark has no Insert fixture:**  
The insertion mechanism is identical to Stage 1. The Stage 2 benchmark isolates the effect of SIMD acceleration on SpMV throughput. Comparing Stage 2 SpMV results against Stage 1 SpMV results directly quantifies the SIMD speedup on the test machine.

---

## Stage 3 — Insert & SpMV (Layout-Aware CSR)

**Branch:** `stage3/layout-aware-csr`  
**File:** `bench/spira_bench.cpp`

Stage 3 introduces the formal **open / locked lifecycle** and a single flat CSR allocation. The benchmark API changes from `set_mode` / `flush()` to `open()` / `lock()`.

### InsertFixture (Stage 3)

**What is being measured:**  
The cost of `open()` → insert BATCH entries → `lock()` against a matrix at steady-state density. The `lock()` here calls `merge_csr` (two-pointer merge) for dirty rows and skips clean rows via the dirty bitset.

```
SetUp:
  ① fill_and_lock(full triples)   ← builds flat CSR
  ② matrix remains locked

Timed loop:
  PauseTiming → flush_cache → ResumeTiming
  ┌────────────────────────────────────────────────────┐
  │  mat.open()                                        │
  │    ← flip mode flag; buffer ready; CSR preserved  │
  │                                                    │
  │  for each t in batch:                              │
  │      mat.insert(t.row, t.col, t.val)               │
  │      ← O(1) hash-map write + dirty_[row] = true   │
  │                                                    │
  │  mat.lock()                                        │
  │    ← for each row:                                 │
  │        dirty[row]=false → memmove (fast path)      │
  │        dirty[row]=true  → sort+dedup+merge_csr     │
  └────────────────────────────────────────────────────┘
```

**Dirty bitset effect on Insert cost:**

```
BATCH=256 random inserts into N=10 000 rows:
  Expected distinct rows touched ≈ 256 (at 0.1 % density, likely 256 unique rows)
  → dirty fraction ≈ 256 / 10 000 = 2.56 %

lock() cost breakdown:
  9 744 clean rows → single memmove per row (bulk CSR copy, ~2 cache lines each)
    ~9 744 × 128 B = 1.25 MB  total copy  (cheap sequential writes)
  256 dirty rows   → sort + dedup + merge_csr (pays full O(k log k) cost)
    cost dominated by these 256 rows only
```

This is why Stage 3 Insert throughput scales differently with density than Stage 1: the dirty bitset amortises the per-lock merge cost when only a small fraction of rows change.

### SpMVFixture (Stage 3)

Structurally identical to Stage 2 SpMV. The key difference is the locked matrix now uses a single contiguous flat CSR allocation:

```
Stage 1/2: per-row sorted slabs (heap fragments)
  row 0 → slab at 0x7f...A0
  row 1 → slab at 0x7f...B4   ← non-contiguous
  ...

Stage 3: single flat CSR
  offsets: [0, 2, 5, 8, ...]    ← one array, N+1 entries
  cols:    [c₀, c₁, c₂, ...]   ← one array, nnz entries
  vals:    [v₀, v₁, v₂, ...]   ← one array, nnz entries

SpMV row access:
  row i lives at cols[offsets[i] .. offsets[i+1]]
  → sequential prefetch-friendly layout
  → hardware prefetcher can predict next row's data
```

The flat CSR layout is expected to improve SpMV bandwidth utilisation compared to Stage 1/2 because the hardware prefetcher can track sequential access across the entire value array.

---

## Stage 4 — Parallel Insert, SpMV & Thread Scaling

**Branch:** `stage4/MutilThreaded`  
**File:** `bench/spira_bench.cpp`

Stage 4 uses `parallel_matrix` — the matrix is statically partitioned into T row ranges, one per thread. All three benchmarks exercise the parallel path.

### InsertFixture (Stage 4)

**What is being measured:**  
The parallel open/fill/lock cycle: T threads each insert their shard of BATCH entries and lock their own partition's CSR concurrently.

```
SetUp:
  ① T = hardware_concurrency()
  ② PM mat(N, N, T)              ← partition into T row ranges
  ③ fill_and_lock(full triples)  ← steady-state CSR, all partitions
  ④ distribute batch by thread:
       distribute_by_thread(mat, batch, T)
       → binary search on partition start rows
       → each thread gets its own sub-list of the 256 entries

Timed loop:
  PauseTiming → flush_cache → ResumeTiming
  ┌───────────────────────────────────────────────────────────┐
  │  mat.open()                                               │
  │                                                           │
  │  mat.parallel_fill(λ):                                    │
  │    thread 0: inserts batch_by_thread[0] into rows[0..k]  │
  │    thread 1: inserts batch_by_thread[1] into rows[k..m]  │
  │    ...  (all concurrent, no synchronisation needed)       │
  │                                                           │
  │  mat.lock()                                               │
  │    thread 0: merge_csr / build_csr for partition 0       │
  │    thread 1: merge_csr / build_csr for partition 1       │
  │    ...  (all concurrent, each thread owns its CSR)        │
  └───────────────────────────────────────────────────────────┘
```

**Thread pool synchronisation overhead:**

```
parallel_fill() / lock() each call pool.execute(fn):
  main thread:  ① increment generation counter
                ② notify_all() on start condition variable
                ③ block on done semaphore

  each worker:  ① wakes on generation change
                ② executes fn(tid)
                ③ decrements finished counter (atomic)
                ④ last worker signals done semaphore

Total overhead per execute():
  1× mutex lock + N× condition variable wake + 1× semaphore signal
  ≈ 1–5 µs on Linux (negligible vs. the lock/fill cost at 1 % density)
```

**Why pre-distribute the batch?**  
`distribute_by_thread` runs outside the timed loop. In a real application, data partitioning is typically done once per timestep and amortised; the benchmark isolates the insert + merge cost, which is the performance-critical section.

### SpMVFixture (Stage 4)

**What is being measured:**  
`parallel::algorithms::spmv(mat, x, y)` — each thread computes `y[r_start..r_end] = A[r_start..r_end] × x` for its own partition, using the SIMD kernel from Stage 2.

```
SetUp:
  T = hardware_concurrency()
  PM mat(N, N, T) — NNZ-balanced partitioning:

    Row NNZ:   [ 2, 8, 1, 5, 3, 9, 2, 4, 6, 1, ... ]
               ─────────────────────────────────────
    Partition boundaries chosen via prefix-sum so each
    thread handles ≈ total_nnz / T non-zeros.
    → threads finish at roughly the same time (load balance)

Timed loop:
  PauseTiming → flush_cache → ResumeTiming
  ┌────────────────────────────────────────────────────────┐
  │  parallel::spmv(mat, x, y)                            │
  │                                                        │
  │    thread 0 → rows[0..k]:   y[i] += SIMD dot(row_i)  │
  │    thread 1 → rows[k..m]:   y[i] += SIMD dot(row_i)  │
  │    ...                                                 │
  │    (all concurrent; x is read-only shared, y ranges   │
  │     are disjoint — no false sharing, no locks)        │
  └────────────────────────────────────────────────────────┘
```

**Parallelism model:**  
SpMV is embarrassingly parallel at the row level because output rows are independent. The only shared state is the read-only input vector `x`. At low density `x` fits in LLC and is effectively broadcast; at high density `x` is also DRAM-bound and accessed by all threads simultaneously, increasing memory bus pressure.

### ThreadScalingFixture (Stage 4)

**What is being measured:**  
A **strong-scaling study**: the same problem (N=10 000, 1 % density, random pattern) solved with varying thread counts T ∈ {1, 2, 4, 8, hardware_concurrency}.

```
For each T in {1, 2, 4, 8, hw_concurrency}:
  SetUp:
    PM mat(N, N, T)   ← NNZ-balanced partition for T threads
    fill_and_lock(100 nnz/row, random)

  Timed loop:
    PauseTiming → flush_cache → ResumeTiming
    parallel::spmv(mat, x, y)   ← T threads active
```

**Why 1 % density?**  
At 1 % the working set is ~12 MB. This is:
- Above a typical single-core L3 slice (~4–6 MB) so it is not entirely cache-resident for one thread.
- Below the total LLC of an 8-core machine (~32–48 MB) so multi-core runs can share the LLC.
- This places the benchmark at the **memory-bandwidth crossover** — the most informative point for a scaling study.

**Interpreting the scaling results:**

```
Ideal strong scaling:
  time(T) = time(1) / T
  speedup(T) = T

Memory-bus saturation (Amdahl + bandwidth):
  If N threads together exceed DRAM bandwidth B:
    speedup(T) flattens at   T_sat = B / bandwidth_per_thread

Amdahl's Law (serial fraction s):
  speedup(T) ≤ 1 / (s + (1-s)/T)
  → fit s from the {1, 2, 4, 8} series to estimate serial bottleneck

Expected shape on r8i.4xlarge (8 physical cores, ~100 GB/s DRAM):
  T=1:   baseline
  T=2:   ~1.9× (near-ideal, LLC likely shared)
  T=4:   ~3.5× (some bandwidth contention)
  T=8:   ~5–6× (memory bus partially saturated)
```

---

## Runner Script

`bench/run_benchmark.sh` automates the full build-and-run pipeline for all four stages on a Linux server.

```
./bench/run_benchmark.sh [--no-smt-disable]
```

**Execution flow:**

```
1. Detect CPU topology
   lscpu -p=cpu,core → map physical_core_id → first logical CPU
   Identifies which CPUs are HT siblings vs. physical cores.

2. Optionally disable SMT (requires root)
   echo off > /sys/devices/system/cpu/smt/control
   Re-reads CPU list after disabling.
   Restored on EXIT via trap.

3. Build each stage in a git worktree
   .bench_worktrees/
   ├── stage1/  git worktree of stage1/MVP
   ├── stage2/  git worktree of stage2/SIMD-Kernels
   ├── stage3/  git worktree of stage3/layout-aware-csr
   └── stage4/  git worktree of stage4/MutilThreaded
   Each built with: cmake -DCMAKE_BUILD_TYPE=Release -DSPIRA_BUILD_BENCHMARKS=ON

4. Generate run_all.sh
   Sequential runner: stage1 → stage2 → stage3 → stage4
   CPU pinning via taskset:
     stages 1–3 (serial): taskset -c <single_physical_core>
     stage 4 (parallel):  taskset -c <all_physical_cores>

5. Launch tmux session 'spira_bench'
   Single window, all stages sequential.
   Output tee'd to bench_results/results_<timestamp>.txt
   JSON also saved per stage for machine parsing.
```

**CPU pinning rationale:**

```
Serial benchmarks pinned to one core:
  → eliminates OS scheduler migration between cores
  → ensures consistent LLC and DRAM latency
  → results reflect single-core peak performance

Parallel benchmark uses all physical cores:
  → SMT disabled (or HT siblings excluded) to prevent
    two benchmark threads sharing one physical core's
    execution units and distorting scaling results
  → taskset prevents the OS from scheduling on the
    leftover cores not in the affinity mask
```

**Retrieving results:**

```bash
scp ubuntu@<instance-ip>:~/spira/bench_results/results_<timestamp>.txt .
scp ubuntu@<instance-ip>:~/spira/bench_results/results_<timestamp>_stage*.json .
```

**Google Benchmark flags used:**

| Flag | Value | Reason |
|------|-------|--------|
| `--benchmark_repetitions` | 5 | Run each case 5 times; report mean, median, stddev. |
| `--benchmark_report_aggregates_only` | true | Suppress per-repetition rows; show only mean/median/stddev. |
| `--benchmark_out_format` | json | Machine-readable output for post-processing or plotting. |
| `--benchmark_format` | console | Human-readable table on stdout (also tee'd to .txt). |

---

## Interpreting Results

### Comparing Across Stages

Each stage adds capabilities that should show measurable performance changes:

| Transition | Expected effect on SpMV |
|---|---|
| Stage 1 → Stage 2 | Throughput increases (FLOP/s and GB/s) due to SIMD vectorisation. Magnitude depends on ISA: AVX2 ≈ 2–3× scalar; AVX-512 ≈ 3–4×. |
| Stage 2 → Stage 3 | Flat CSR improves prefetch efficiency. May see moderate GB/s improvement at medium density where the hardware prefetcher can track sequential access. |
| Stage 3 → Stage 4 | Near-linear strong scaling up to the memory bandwidth limit. Expect 5–7× at 8 physical cores for 1 % density on r8i.4xlarge. |

### Roofline Position

Given the bandwidth model, estimate the roofline position for SpMV:

```
Arithmetic intensity (AI) = FLOPs / bytes
  = 2 × nnz / (nnz × 12 + N × 16 + (N+1) × 8)

At 1 % density (nnz=1 000 000, N=10 000):
  AI ≈ 2 000 000 / 12 240 008 ≈ 0.163 FLOP/B

r8i.4xlarge peak DRAM bandwidth ≈ 100 GB/s  (measured via STREAM)
r8i.4xlarge peak AVX-512 FLOP/s ≈ 200 GFLOP/s  (8 cores × 25 GFLOP/s)

Roofline peak for SpMV:
  min(100 GB/s × 0.163 FLOP/B,  200 GFLOP/s)
  = min(16.3 GFLOP/s, 200 GFLOP/s)
  = 16.3 GFLOP/s    ← memory-bound for all realistic thread counts

Achieved GFLOP/s from benchmark:
  items_per_second / 1e9  ≈ roofline efficiency × 16.3

Achieved GB/s from benchmark:
  bytes_per_second / 1e9  ≈ DRAM bandwidth utilisation × 100
```

### Insert Throughput Interpretation

Insert throughput (items/sec) is governed by two competing costs:

```
Low density (10 nnz/row):
  - merge_csr work is cheap (short sorted runs)
  - Most cost is buffer hash-map overhead + random insert addressing
  - Throughput limited by hash-map write latency

High density (1000 nnz/row):
  - merge_csr work is expensive (long CSR rows, large working set)
  - Dirty bitset helps: BATCH=256 entries spread across 10 000 rows
    means only ~256 rows are dirty regardless of density
  - Throughput limited by merge_csr for dirty rows + CSR copy for clean rows
  - Flat CSR (Stage 3+) improves the copy path over Stage 1's fragmented slabs
```
