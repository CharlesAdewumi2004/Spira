cmake_minimum_required(VERSION 3.22)
project(spira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# ---------- Options ----------
option(SPIRA_BUILD_TESTS       "Build unit tests"           ON)
option(SPIRA_BUILD_BENCHMARKS  "Build performance benches"  ON)
option(SPIRA_ENABLE_SIMD       "Build SIMD-accelerated kernels" ON)

# We'll use FetchContent for deps
include(FetchContent)

# -------------------------------------------------------------------
# Boundcraft (dependency)
# IMPORTANT: force its tests/bench OFF when consumed via FetchContent
# -------------------------------------------------------------------
set(BOUNDCRAFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BOUNDCRAFT_BUILD_BENCH OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  boundcraft_repo
  GIT_REPOSITORY https://github.com/CharlesAdewumi2004/Boundcraft.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(boundcraft_repo)

if (TARGET boundcraft::boundcraft)
  set(_BOUNDCRAFT_TARGET boundcraft::boundcraft)
elseif (TARGET boundcraft)
  set(_BOUNDCRAFT_TARGET boundcraft)
else()
  message(FATAL_ERROR "Boundcraft did not define a target (expected boundcraft::boundcraft or boundcraft).")
endif()

# ---------- unordered_dense (ankerl::unordered_dense) ----------
FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG        v4.5.0
)
FetchContent_MakeAvailable(unordered_dense)

# ---------- Library (header-only core) ----------
add_library(spira INTERFACE)
target_include_directories(spira INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(spira INTERFACE cxx_std_23)

# Make deps available to all spira consumers
target_link_libraries(spira INTERFACE unordered_dense::unordered_dense ${_BOUNDCRAFT_TARGET})

# ---------- SIMD Kernel Library ----------
if (SPIRA_ENABLE_SIMD)
  set(KERNEL_SOURCES
    src/kernels/dispatch.cpp
    src/kernels/dot_impls/dot_scalar.cpp
  )

   # x86 SIMD sources — only on x86 architectures
   if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
        list(APPEND KERNEL_SOURCES
        src/kernels/dot_impls/dot_scalar.cpp
    )
  endif()

#   # ARM SIMD sources — only on ARM architectures
#   if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|armv[78]")
#     list(APPEND KERNEL_SOURCES
#       src/kernels/dot_neon.cpp
#     )
#   endif()

  add_library(spira_kernels STATIC ${KERNEL_SOURCES})
  target_include_directories(spira_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_compile_features(spira_kernels PUBLIC cxx_std_23)

  # Per-file SIMD compile flags (x86 — GCC/Clang)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      set_source_files_properties(src/kernels/dot_sse.cpp
        PROPERTIES COMPILE_OPTIONS "-msse4.2"
      )
      set_source_files_properties(src/kernels/dot_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
      )
      set_source_files_properties(src/kernels/dot_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vl;-mavx512bw;-mavx512dq"
      )
    elseif (MSVC)
      set_source_files_properties(src/kernels/dot_sse.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:SSE2"
      )
      set_source_files_properties(src/kernels/dot_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
      )
      set_source_files_properties(src/kernels/dot_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
      )
    endif()
  endif()

  # ARM: NEON is enabled by default on aarch64, no special flags needed

  # Link kernels into spira
  target_link_libraries(spira INTERFACE spira_kernels)
endif()

# ---------- Perf-friendly flags for Debug / RelWithDebInfo ----------
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer)
endif()

# ---------- Tests ----------
if (SPIRA_BUILD_TESTS)
  # Make sure no stale cache forces a local googletest path
  unset(FETCHCONTENT_SOURCE_DIR_GOOGLETEST CACHE)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "" CACHE PATH "" FORCE)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(unit_tests
    tests/layout_and_type_constraits_tests.cpp
    tests/matrix_tests.cpp
    tests/flush_behaviour_tests.cpp
    tests/mode_switch_tests.cpp
    tests/spmv_tests.cpp
    tests/large_matrix_stress_tests.cpp
    tests/spgemm_tests.cpp
    tests/transpose_tests.cpp
    tests/matrix_addition_tests.cpp
    tests/scaler_tests.cpp
    tests/matrix_operator_overloads_tests.cpp
  )

  target_link_libraries(unit_tests PRIVATE spira GTest::gtest_main)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(unit_tests PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(unit_tests PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  endif()

  if (UNIX AND NOT APPLE)
    target_link_libraries(unit_tests PRIVATE pthread)
  endif()

  include(GoogleTest)
  gtest_discover_tests(unit_tests)
endif()

# ---------- Benchmarks ----------
if (SPIRA_BUILD_BENCHMARKS)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(GB_spira
    bench/google-benchmark/spira_benchmark.cpp
  )

  target_link_libraries(GB_spira PRIVATE spira benchmark::benchmark)

  if (UNIX AND NOT APPLE)
    target_link_libraries(GB_spira PRIVATE pthread)
  endif()

  target_compile_features(GB_spira PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(GB_spira PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(GB_spira PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(GB_spira PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  endif()
endif()