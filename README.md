# Spira Documentation

Spira is a header only C++23 library for dynamic sparse matrices. The goal of
the project is to explore how to build a sparse container that can be cheaply
mutated, cheaply read, and vectorised on modern hardware, without forcing a
user to pick between those three at construction time.

This folder holds the reference documentation for the library. If you just
want a high level tour, start with the overview. If you want to dig into why
things are laid out the way they are, go to architecture. The PlantUML files
under `diagrams/` are the source of truth for every diagram referenced below.

## Reading order

1. [overview.md](overview.md) describes the project, the four development
   stages, and the core ideas that shape the API.
2. [architecture.md](architecture.md) walks through every layer of the
   library, from the concept constraints at the bottom up to parallel SpMV
   at the top. Each section points at the PlantUML diagram that goes with it.
3. [api-reference.md](api-reference.md) lists the public API surface with
   type signatures and short examples.
4. [build-and-test.md](build-and-test.md) covers CMake options, the test
   layout, and how the benchmark harness works.

## Diagram index

All diagrams live under [diagrams/](diagrams/) as PlantUML source. Render them
with the PlantUML CLI, a PlantUML server, or an IDE plugin. Each diagram is
self contained so you can read it in isolation.

| File | Type | What it shows |
|------|------|---------------|
| [01-class-diagram.puml](diagrams/01-class-diagram.puml) | Class | Core matrix, row, partition, and CSR storage types with their template parameters |
| [02-state-lifecycle.puml](diagrams/02-state-lifecycle.puml) | State | The open / locked lifecycle and the transitions between them |
| [03-lock-sequence.puml](diagrams/03-lock-sequence.puml) | Sequence | What happens inside a single call to `matrix::lock()` |
| [04-spmv-sequence.puml](diagrams/04-spmv-sequence.puml) | Sequence | How a parallel SpMV dispatches down to a SIMD kernel |
| [05-component-layers.puml](diagrams/05-component-layers.puml) | Component | The dependency stack from concepts to parallel algorithms |
| [06-partition-layout.puml](diagrams/06-partition-layout.puml) | Object | How `parallel_matrix` splits rows into per thread partitions |
| [07-csr-build-activity.puml](diagrams/07-csr-build-activity.puml) | Activity | The two pointer merge used when rebuilding CSR after re-lock |
| [08-kernel-dispatch.puml](diagrams/08-kernel-dispatch.puml) | Activity | Startup time kernel selection based on CPU feature detection |
| [09-buffer-hierarchy.puml](diagrams/09-buffer-hierarchy.puml) | Class | The CRTP buffer interface and its three implementations |
| [10-thread-pool.puml](diagrams/10-thread-pool.puml) | Sequence | The generation counter protocol between main thread and workers |
| [11-memory-layout.puml](diagrams/11-memory-layout.puml) | Object | AoS vs SoA CSR memory layout side by side |
| [12-deployment.puml](diagrams/12-deployment.puml) | Deployment | How the header library, kernel library, tests, and benchmarks link together |

## Status and scope

The current active branch is `stage4/MutilThreaded`. This branch contains the
full four stage implementation: the mutable matrix, the SIMD kernel dispatch
layer, the compressed storage with open and locked modes, and the partitioned
multi threaded matrix. Other branches in the repository hold earlier or
experimental stages and are documented in [STAGES.md](../STAGES.md) at the
project root.

The documentation in this folder tracks the current branch. When you see a
reference to a specific file or header it is relative to the project root.
