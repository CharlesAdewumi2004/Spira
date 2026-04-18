# Build, test, and benchmark

Spira uses CMake 3.22 or newer and a C++23 compiler. It has been tested with
GCC 13, Clang 17, and MSVC 19.37. Earlier compilers may work but are not
part of CI.

## Dependencies

All external dependencies are fetched automatically by CMake via
`FetchContent`. There is nothing to install manually.

| Dependency | Version | Purpose |
|------------|---------|---------|
| `boundcraft` | latest | Hybrid binary/linear searcher used inside buffers |
| `ankerl::unordered_dense` | 4.5.0 | Fast hash map for the per row buffer index |
| GoogleTest | 1.14.0 | Unit test framework (optional) |
| Google Benchmark | 1.8.4 | Benchmark framework (optional) |

The standard library requirements come from C++23: `std::jthread`,
`std::binary_semaphore`, `std::move_only_function`, and `std::aligned_alloc`
are all used unconditionally.

## Build options

The top level `CMakeLists.txt` exposes the following options:

| Option | Default | Effect |
|--------|---------|--------|
| `SPIRA_BUILD_TESTS` | `ON` | Build the `unit_tests` executable |
| `SPIRA_BUILD_PARALLEL` | `ON` | Enable the parallel matrix, thread pool, and parallel algorithms, plus the `parallel_tests` executable |
| `SPIRA_BUILD_BENCHMARKS` | `ON` | Build the `spira_bench` executable |
| `SPIRA_ENABLE_SIMD` | `ON` | Compile the `spira_kernels` static library with SSE, AVX, AVX-512, and NEON kernels |

The library target `spira` is always header only. When `SPIRA_ENABLE_SIMD`
is on it links against the `spira_kernels` static library; when off, the
SIMD SpMV specialisation is compiled out and all SpMV calls go through the
scalar path.

## Configuring and building

Debug build with sanitisers enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

Release build, optimised for benchmarking:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
```

A typical Release build uses `-O3 -march=native -DNDEBUG`. The benchmark
target adds `-march=native` unconditionally because it needs the SIMD
instructions that match the local CPU.

### SIMD compilation details

The SIMD kernel files live in `src/kernels/dot_impls/x86/` and
`src/kernels/dot_impls/arm/`. Each kernel file needs its own compile
flags so the compiler can emit the right instructions without enabling
them globally:

| Kernel | File | Compile flags |
|--------|------|---------------|
| Scalar fallback | `dot_scalar.cpp` | none (always compiles) |
| SSE 4.2 | `x86/dot_sse.cpp` | `-msse4.2` |
| AVX2 with FMA | `x86/dot_avx.cpp` | `-mavx2 -mfma` |
| AVX-512 | `x86/dot_avx512.cpp` | `-mavx512f -mavx512vl -mavx512bw -mavx512dq` |
| NEON | `arm/dot_neon.cpp` | (built in on AArch64) |

The per file flags are applied by CMake via `set_source_files_properties`
in the kernel CMakeLists. The x86 kernels are only compiled when the
detected architecture is `x86`, `x86_64`, or `AMD64`. The NEON kernel is
only compiled when the architecture is `aarch64` or `arm*`. The scalar
fallback is always compiled so every build has a working dispatch
target.

Because the dispatch layer selects the best available kernel at runtime,
you can build the library with all four kernels present and ship a
single binary that works everywhere. The only machine specific part is
the compile time enable of the kernel file itself.

## Running tests

```bash
ctest --test-dir build --output-on-failure
```

The test suite is split into two CTest targets:

- `unit_tests` covers the core matrix, concepts, buffers, CSR storage,
  operator overloads, the SIMD kernels themselves, and the serial
  algorithms.
- `parallel_tests` covers the thread pool, partition, parallel matrix,
  and parallel algorithms. Only built when `SPIRA_BUILD_PARALLEL=ON`.

Both executables are linked against GoogleTest and can be run directly:

```bash
./build/tests/unit_tests
./build/tests/parallel_tests
```

The `simd_dot_operator_tests` file is by far the largest test file in
the suite. It runs the same dot product problem through every kernel
(scalar, SSE, AVX, AVX-512, NEON) and cross checks the results against a
ground truth computed in double precision. It is what catches kernel
regressions when intrinsics are rearranged.

## Running benchmarks

Benchmarks live in `bench/spira_bench.cpp` and are built as the
`spira_bench` executable when `SPIRA_BUILD_BENCHMARKS=ON`.

```bash
cmake --build build-release --target spira_bench
./build-release/bench/spira_bench
```

Google Benchmark flags work as usual:

```bash
# filter
./build-release/bench/spira_bench --benchmark_filter=SpMV

# output to json
./build-release/bench/spira_bench --benchmark_format=json > results.json

# repetitions
./build-release/bench/spira_bench --benchmark_repetitions=5 \
                                  --benchmark_report_aggregates_only=true
```

The benchmark suite has three fixtures:

- `InsertFixture` measures the cost of a full open, insert batch, lock
  cycle on an already populated matrix. The timed region includes the
  lock.
- `SpMVFixture` measures a single SpMV on a locked matrix with a cold
  last level cache. It flushes the LLC between iterations by striding
  through a scratch buffer.
- `ThreadScalingFixture` varies the thread count from 1 up to
  `hardware_concurrency()` on a fixed problem to measure strong
  scaling.

All three report both items per second (interpretable as FLOP/s for
SpMV) and bytes per second (effective bandwidth). The methodology is
spelled out in [`../BENCHMARKS.md`](../BENCHMARKS.md) at the project
root.

## Repository layout

```
Spira/
  CMakeLists.txt
  README.md
  STAGES.md
  BENCHMARKS.md
  LICENSE
  bench/
    spira_bench.cpp            benchmark driver
  cmake/
    ...                        helper modules
  include/
    spira/
      spira.hpp                umbrella include
      config.hpp
      concepts.hpp
      traits.hpp
      matrix/
        matrix.hpp
        matrix_operators.hpp
        row.hpp
        buffer/
          buffer_base.hpp
          buffer_tags.hpp
          buffer_tag_traits.hpp
          aos_array_buffer.hpp
          soa_array_buffer.hpp
          hash_map_buffer.hpp
        layout/
          layout_tags.hpp
          element_pair.hpp
        storage/
          csr_storage.hpp
          csr_storage_soa.hpp
          csr_storage_aos.hpp
          csr_storage_detail.hpp
          csr_build.hpp
      serial/
        serial.hpp
        spmv.hpp
        spgemm.hpp
        transpose.hpp
        matrix_addition.hpp
        accumulate.hpp
        scalars.hpp
      parallel/
        parallel.hpp
        parallel_matrix.hpp
        partition.hpp
        thread_pool.hpp
        insert_staging.hpp
        algorithms/
          spmv.hpp
          spgemm.hpp
          transpose.hpp
          matrix_addition.hpp
          accumulate.hpp
          scalars.hpp
  src/
    kernels/
      dispatch.cpp               startup dispatch + function pointers
      hw_detect.hpp              CPU feature detection
      runtime_config.hpp         singleton with CpuFeatures + latency
      simd_aliases.hpp
      simd_aliases/              per architecture intrinsic wrappers
      dot_impls/
        dot_scalar.cpp
        x86/
          dot_sse.cpp
          dot_avx.cpp
          dot_avx512.cpp
        arm/
          dot_neon.cpp
  tests/
    ...                          GoogleTest files
  docs/
    README.md                    this documentation set
    overview.md
    architecture.md
    api-reference.md
    build-and-test.md
    diagrams/
      01-class-diagram.puml
      ...
```

## Rendering the PlantUML diagrams

The diagrams under `docs/diagrams/` are plain PlantUML source. You can
render them in a few ways.

Using the PlantUML CLI:

```bash
plantuml docs/diagrams/*.puml
```

This produces `.png` files next to the `.puml` source. Pass
`-tsvg` for SVG output.

Using a local server with the Docker image:

```bash
docker run --rm -v "$PWD/docs/diagrams":/puml plantuml/plantuml \
           -tsvg /puml/*.puml
```

Using VSCode: install the PlantUML extension, open any `.puml` file,
and use the preview pane.

Using IntelliJ or CLion: the PlantUML integration plugin renders the
preview automatically when a `.puml` file is open.
