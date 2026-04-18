# Overview

Spira is a dissertation project that asks a fairly narrow question: if you
need a sparse matrix that you can both fill in incrementally *and* use for
high throughput numerical kernels, what does a good design actually look
like? Most production sparse libraries pick one of the two. BLAS style
libraries assume you hand them an already built CSR array. Graph or assembly
libraries keep everything in hash maps and accept the penalty on every read.
Spira tries to keep both worlds in the same container by letting the user
control when the structure freezes.

The code is C++23 and header only for almost everything. The one exception
is a small static library of SIMD kernels whose compilation flags have to be
set per file because you cannot mix AVX-512 and SSE in the same translation
unit on most compilers without upsetting the target machine at load time.

## What the library gives you

At the top level there are two containers.

`spira::matrix` is the single threaded matrix. You construct it with a row
and column count, you call `insert(row, col, value)` as many times as you
want, and then you call `lock()` to freeze it. From that point on reads go
through a flat CSR array and are as fast as a hand rolled CSR library. If
you want to edit the matrix again, you call `open()` and the buffer machinery
comes back online. CSR built by the previous lock is kept around so repeated
cycles are cheap.

`spira::parallel::parallel_matrix` is the multi threaded sibling. It owns a
fixed size thread pool and a vector of partitions. Each partition holds a
slice of the row range and its own CSR storage. Locking and reading the
matrix in parallel is just the pool dispatching the serial version of the
operation onto each partition. Because partitions are disjoint there is no
synchronisation on the critical path.

Both containers are templated on a layout tag, an index type, a value type,
a buffer tag, an initial reserve hint, and a lock policy. The parallel one
adds an insert policy and a staging buffer size on top. All of those choices
are made at compile time so the compiler can specialise the read and SpMV
paths aggressively.

## The four stages

Spira was built in four stages that roughly track a dissertation chapter
each. Every stage lives on its own branch. The current working branch
(`stage4/MutilThreaded`) contains all four stacked on top of each other.

**Stage 1, the MVP.** A dynamic sparse matrix with a two layer per row
storage: an unsorted write buffer and a sorted slab. Inserts go to the
buffer, reads flush the buffer into the slab on first touch. Three matrix
modes (`spmv`, `balanced`, `insert_heavy`) pick different buffer sizes for
different access patterns. This stage proves out the basic abstraction.

**Stage 2, SIMD kernels and hardware detection.** Adds a dispatch layer
that probes the CPU at startup, measures DRAM latency with a pointer chase
chain, and installs function pointers pointing at the best available
sparse dot product kernel. SSE, AVX2 with FMA, AVX-512, and NEON are all
present, with a scalar fallback. The hot SpMV path calls the function
pointer directly, so there are no branches on the inner loop.

**Stage 3, layout aware CSR.** Retires the three mode system and replaces
it with an explicit `open` / `locked` lifecycle. Locked mode builds a flat
CSR array with 64 byte aligned storage. An AoS layout keeps each column
index next to its value. A SoA layout keeps columns and values in separate
arrays. Repeated lock cycles use a two pointer merge so you only pay for
the rows that changed, tracked with a dirty bitset.

**Stage 4, multi threading.** Adds the parallel matrix, the thread pool,
and parallel overloads for every algorithm. Rows are assigned to threads
using an NNZ balanced split so work is distributed evenly even when the
matrix has highly non uniform density. An optional staged insert policy
burst flushes inserts from a small per partition staging array to keep
the hot write path in L1.

The progression is additive. Nothing from stage 1 was thrown away, it just
got more sophisticated at each step. See [STAGES.md](../STAGES.md) at the
project root for the full stage level design notes with ASCII diagrams.

## Core ideas to internalise

There are four ideas worth having in your head before reading the rest of
the documentation.

**The open / locked split is the central design choice.** In open mode the
matrix is cheap to mutate and slow to read. In locked mode it is impossible
to mutate and very cheap to read. Every other design decision in the
library follows from making this split cheap and predictable.

**Layout is a compile time tag, not a runtime flag.** Choosing between AoS
and SoA is a template parameter, which means the compiler generates a
different class for each choice. This is necessary because the SIMD SpMV
kernel wants to load columns and values as separate vectors (SoA), while
scalar access patterns benefit from the column and value sitting on the
same cache line (AoS).

**The SIMD dispatch is centralised and runs once.** A global static
initialiser inspects the CPU, picks the best kernel, and stores a function
pointer. The hot path calls that function pointer. You do not pay a
branch on each row, and you do not pay the cost of `std::function`.

**Parallelism is partition based, not work stealing.** The parallel matrix
divides rows into fixed partitions at construction. Each thread owns its
partition for the life of the matrix. Locking, reading, and SpMV are
embarrassingly parallel because partitions do not share state. Re-balancing
after density drift is an explicit operation the user calls when they want
it.

## Who this is for

The library targets people who are already comfortable writing numerical
C++ and want to experiment with alternative layouts, hand tuned kernels,
or threading strategies. It is not a drop in replacement for Eigen,
SuiteSparse, or Intel MKL. It is small enough to read end to end in an
afternoon and is meant to be forked and modified.

If you are new to sparse matrices, it is probably worth reading a short
introduction to CSR first, for example the SciPy sparse documentation or
the Saad book. Once you know what CSR is, the rest of Spira will make
sense quickly.
