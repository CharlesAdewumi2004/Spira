# API Reference

This is the complete public API surface of Spira. Everything documented here
is in the `spira`, `spira::serial::algorithms`, `spira::parallel`, or
`spira::parallel::algorithms` namespace. The unqualified name in each
section header tells you which namespace to expect.

The header file `<spira/spira.hpp>` pulls in everything. If you only need a
subset, include the specific headers in the relevant subsection below.

## Table of contents

1. [Concepts and traits](#concepts-and-traits)
2. [Configuration enumerations](#configuration-enumerations)
3. [Layout tags](#layout-tags)
4. [Buffer tags](#buffer-tags)
5. [`matrix`](#matrix)
6. [Operator overloads](#operator-overloads)
7. [Serial algorithms](#serial-algorithms)
8. [`parallel_matrix`](#parallel_matrix)
9. [Parallel algorithms](#parallel-algorithms)
10. [`thread_pool`](#thread_pool)
11. [CSR storage and slices](#csr-storage-and-slices)
12. [SIMD kernels](#simd-kernels)

## Concepts and traits

Header: `<spira/concepts.hpp>`, `<spira/traits.hpp>`.

```cpp
namespace spira::concepts {

    // Column index type: unsigned integral of at least 32 bits.
    // Rejects bool and character types.
    template <typename I>
    concept Indexable = /* unsigned integral && sizeof >= 4 */;

    // Value type: arithmetic, default constructible, equality comparable.
    // Must provide ValueTraits<V>::zero() and ValueTraits<V>::is_zero(x).
    template <class V>
    concept Valueable = /* ... */;
}

namespace spira::traits {

    template <class V>
    struct ValueTraits {
        static constexpr V zero() noexcept;
        static constexpr bool is_zero(V x) noexcept;
        static constexpr bool is_zero(V x, V eps) noexcept; // float only
    };

    template <class V>
    using AccumulationOf = /* widened type for safe accumulation */;
}
```

Specialisations exist for built in arithmetic types, `std::complex<T>`, and
any user type that satisfies the `Valueable` contract.

## Configuration enumerations

Header: `<spira/config.hpp>`.

```cpp
namespace spira::config {

    enum class matrix_mode : uint8_t {
        open,
        locked
    };

    enum class lock_policy : uint8_t {
        no_compact,       // never build CSR, use sorted buffers in locked mode
        compact_preserve, // build CSR, keep buffers (default)
        compact_move      // build CSR, free buffers
    };

    enum class insert_policy : uint8_t {
        direct, // write inserts straight to partition row buffers
        staged  // accumulate into per-partition staging arrays first
    };
}
```

## Layout tags

Header: `<spira/matrix/layout/layout_tags.hpp>`.

```cpp
namespace spira::layout::tags {
    struct aos_tag {};
    struct soa_tag {};
}

namespace spira::layout {
    template <class T>
    concept ValidLayoutTag = /* aos_tag or soa_tag */;
}
```

The `elementPair` record used by the AoS path is defined in
`<spira/matrix/layout/element_pair.hpp>`:

```cpp
namespace spira::layout {

    template <concepts::Indexable I, concepts::Valueable V>
    struct elementPair {
        I column{};
        V value{};
    };
}
```

## Buffer tags

Header: `<spira/matrix/buffer/buffer_tags.hpp>`.

```cpp
namespace spira::buffer::tags {

    template <class LayoutPolicy>
    struct array_buffer {
        using layout_policy = LayoutPolicy;
    };

    struct hash_map_buffer {};
}
```

A buffer tag is resolved to a concrete buffer type via the traits template
in `<spira/matrix/buffer/buffer_tag_traits.hpp>`:

```cpp
namespace spira::buffer::traits {

    template <class BufferTag, class I, class V, std::size_t N>
    using traits_of_type = /* concrete buffer<I, V, N> */;
}
```

## `matrix`

Header: `<spira/matrix/matrix.hpp>`. Namespace `spira`.

```cpp
template <
    class LayoutTag,
    concepts::Indexable   I       = uint32_t,
    concepts::Valueable   V       = double,
    class                 BufferTag
        = buffer::tags::array_buffer<layout::tags::aos_tag>,
    std::size_t           BufferN = 64,
    config::lock_policy   LP      = config::lock_policy::compact_preserve
>
class matrix;
```

### Member types

```cpp
using index_type   = I;
using value_type   = V;
using storage_type = row<LayoutTag, I, V, BufferTag, BufferN>;
using size_type    = std::size_t;
using shape_type   = std::pair<size_type, size_type>;
```

### Construction

```cpp
explicit matrix(size_type row_limit, size_type column_limit);

matrix(size_type row_limit, size_type column_limit,
       size_type reserve_per_row);
```

Copy and move constructors and assignment are defaulted. The matrix is
default destructible.

### Shape queries

```cpp
[[nodiscard]] shape_type shape() const noexcept;
[[nodiscard]] size_type  n_rows() const noexcept;
[[nodiscard]] size_type  n_cols() const noexcept;
```

### Mode transitions

```cpp
[[nodiscard]] config::matrix_mode mode() const noexcept;
[[nodiscard]] bool is_locked() const noexcept;
[[nodiscard]] bool is_open()   const noexcept;

void lock();
void open();
```

`lock()` sorts and deduplicates every row buffer, builds or merges the CSR,
and installs per row CSR slices. `open()` transitions back to mutable mode;
the CSR array is kept regardless of policy, but the row buffers may be
re-allocated depending on the lock policy.

### Element operations

```cpp
void insert(size_type row, index_type col, value_type value);

[[nodiscard]] value_type get(size_type row, index_type col) const;
[[nodiscard]] bool       contains(size_type row, index_type col) const;
[[nodiscard]] size_type  row_nnz(size_type row) const;
```

`insert` is only valid in open mode. `get` returns `ValueTraits<V>::zero()`
for missing entries. `row_nnz` returns the current non zero count for the
given row.

### CSR access

```cpp
const csr_storage<LayoutTag, I, V> *csr() const noexcept;

void load_csr(csr_storage<LayoutTag, I, V> &&csr);
```

`csr()` returns a pointer to the flat CSR array if the matrix is locked and
the lock policy built one; `nullptr` otherwise. `load_csr` installs a CSR
built externally and puts the matrix into locked mode without going through
the lock pipeline.

### Iteration

```cpp
template <class Func>
void for_each_row(Func &&f) const;

template <class Func>
void for_each_nnz_row(Func &&f) const;
```

The callback for `for_each_row` receives `(const row_type &row, size_type i)`
for every row, including empty ones. `for_each_nnz_row` skips rows with
zero non zeros.

## Operator overloads

Header: `<spira/matrix/matrix_operators.hpp>`.

```cpp
namespace spira {

    matrix operator+(const matrix &, const matrix &);
    matrix operator-(const matrix &, const matrix &);
    matrix operator*(const matrix &, const matrix &); // SpGEMM

    std::vector<V> operator*(const matrix &, const std::vector<V> &); // SpMV
    matrix operator*(const matrix &, V);                              // scale
    matrix operator/(const matrix &, V);

    matrix operator~(const matrix &); // transpose

    matrix &operator+=(matrix &, const matrix &);
    matrix &operator-=(matrix &, const matrix &);
    matrix &operator*=(matrix &, V);
    matrix &operator/=(matrix &, V);
}
```

All of these forward to the corresponding function in
`spira::serial::algorithms`.

## Serial algorithms

Headers under `<spira/serial/>`. Namespace `spira::serial::algorithms`.

```cpp
// y = A * x
template <class... MatParams, class V>
void spmv(const matrix<MatParams...> &A,
          const std::vector<V> &x,
          std::vector<V> &y);

// C = A * B
template <class... MatParams>
matrix<MatParams...> spgemm(const matrix<MatParams...> &A,
                            const matrix<MatParams...> &B);

// A^T
template <class... MatParams>
matrix<MatParams...> transpose(const matrix<MatParams...> &A);

// A + B
template <class... MatParams>
matrix<MatParams...> matrix_add(const matrix<MatParams...> &A,
                                const matrix<MatParams...> &B);

// sum over row r of A
template <class... MatParams, class Acc = /* AccumulationOf<V> */>
Acc accumulate(const matrix<MatParams...> &A, std::size_t row);

// s * A
template <class... MatParams, class Scalar>
matrix<MatParams...> scale(const matrix<MatParams...> &A, Scalar s);
```

Every function requires its input matrices to be locked. Calling one of
them on an open matrix throws `std::logic_error`.

`spmv` has a specialisation for `soa_tag + uint32_t + float/double` that
calls the SIMD kernel function pointer. All other template parameter
combinations fall through to a scalar per element loop.

## `parallel_matrix`

Header: `<spira/parallel/parallel_matrix.hpp>`. Namespace
`spira::parallel`.

```cpp
template <
    class LayoutTag,
    concepts::Indexable    I        = uint32_t,
    concepts::Valueable    V        = double,
    class                  BufferTag
        = buffer::tags::array_buffer<layout::tags::aos_tag>,
    std::size_t            BufferN  = 64,
    config::lock_policy    LP       = config::lock_policy::compact_preserve,
    config::insert_policy  IP       = config::insert_policy::direct,
    std::size_t            StagingN = 256
>
class parallel_matrix;
```

### Member types

```cpp
using partition_type = partition<LayoutTag, I, V, BufferTag, BufferN, LP>;
using row_type       = typename partition_type::row_type;
using index_type     = I;
using value_type     = V;
using size_type      = std::size_t;
using shape_type     = std::pair<size_type, size_type>;
```

### Construction

```cpp
parallel_matrix(size_type n_rows, size_type n_cols, size_type n_threads);

parallel_matrix(size_type n_rows, size_type n_cols, size_type n_threads,
                size_type reserve_per_row);
```

Copy is deleted; the matrix owns a thread pool. Move is deleted while the
pool is running; see the parallel matrix source for the conditions under
which a move is accepted.

### Shape and concurrency

```cpp
[[nodiscard]] shape_type shape()     const noexcept;
[[nodiscard]] size_type  n_rows()    const noexcept;
[[nodiscard]] size_type  n_cols()    const noexcept;
[[nodiscard]] size_type  n_threads() const noexcept;
```

### Mode transitions

```cpp
[[nodiscard]] config::matrix_mode mode() const noexcept;
[[nodiscard]] bool is_locked() const noexcept;
[[nodiscard]] bool is_open()   const noexcept;

void lock();   // parallel over partitions
void open();   // parallel over partitions
```

### Element operations

```cpp
void insert(size_type row, index_type col, value_type value);

[[nodiscard]] value_type get(size_type row, index_type col)      const;
[[nodiscard]] bool       contains(size_type row, index_type col) const;
[[nodiscard]] size_type  row_nnz(size_type row)                  const;
```

### Parallel fill

```cpp
template <class Func>
void parallel_fill(Func &&fn);
```

`fn` is called once per thread with signature
`fn(std::vector<row_type> &rows, size_type r_start, size_type r_end,
size_type tid)`. The user is responsible for partitioning the input data
by row range.

### Rebalance

```cpp
void rebalance();
```

Recomputes partition boundaries with an nnz balanced split using the
current per row counts and redistributes rows accordingly. Expensive,
only call when the distribution has drifted significantly.

### Partition access

```cpp
const partition_type &partition_at(size_type tid) const;
size_type             n_partitions() const noexcept;
```

Read only access to individual partitions for algorithm implementations.

## Parallel algorithms

Headers under `<spira/parallel/algorithms/>`. Namespace
`spira::parallel::algorithms`.

The function set is identical to the serial set but takes a
`parallel_matrix` instead of a `matrix`:

```cpp
void spmv(const parallel_matrix<...> &A,
          const std::vector<V> &x,
          std::vector<V> &y);

parallel_matrix<...> spgemm   (const parallel_matrix<...> &A,
                               const parallel_matrix<...> &B);

parallel_matrix<...> transpose(const parallel_matrix<...> &A);

parallel_matrix<...> matrix_add(const parallel_matrix<...> &A,
                                const parallel_matrix<...> &B);

value_type           accumulate(const parallel_matrix<...> &A,
                                size_type row);

parallel_matrix<...> scale(const parallel_matrix<...> &A, Scalar s);
```

Each function dispatches the equivalent serial algorithm onto each
partition via the matrix's thread pool. For SpMV and SpGEMM the parallel
path also takes advantage of the SIMD kernel dispatch when the template
parameters match the kernel's contract.

## `thread_pool`

Header: `<spira/parallel/thread_pool.hpp>`. Namespace `spira::parallel`.

```cpp
class thread_pool {
public:
    explicit thread_pool(std::size_t n_threads);
    ~thread_pool();

    thread_pool(const thread_pool &)            = delete;
    thread_pool &operator=(const thread_pool &) = delete;
    thread_pool(thread_pool &&)                 = delete;
    thread_pool &operator=(thread_pool &&)      = delete;

    void execute(std::move_only_function<void(std::size_t)> fn);

    [[nodiscard]] std::size_t size() const noexcept;
};
```

Constructing with zero threads throws `std::invalid_argument`. `execute`
is not re-entrant: a worker must not call `execute` on the same pool
from inside `fn`.

## CSR storage and slices

Headers under `<spira/matrix/storage/>`.

```cpp
namespace spira::storage {

    template <class LayoutTag, class I, class V>
    struct csr_storage; // specialised for aos_tag and soa_tag

    template <class LayoutTag, class I, class V>
    struct csr_slice;   // non owning view into one row of csr_storage
}
```

### SoA specialisation

```cpp
template <class I, class V>
struct csr_storage<layout::tags::soa_tag, I, V> {
    std::size_t n_rows;
    std::size_t nnz;
    std::size_t capacity;

    std::unique_ptr<std::size_t[], /* free */> offsets; // [n_rows + 1]
    /* aligned buffer */                       cols;    // [capacity], 64-byte
    /* aligned buffer */                       vals;    // [capacity], 64-byte
};

template <class I, class V>
struct csr_slice<layout::tags::soa_tag, I, V> {
    const I *cols;
    const V *vals;
    std::size_t nnz;

    const V     *binary_search(I col) const;
    template <class Func>
    void         for_each_element(Func &&f) const;
    V            accumulate() const;
};
```

### AoS specialisation

```cpp
template <class I, class V>
struct csr_storage<layout::tags::aos_tag, I, V> {
    std::size_t n_rows;
    std::size_t nnz;
    std::size_t capacity;

    std::unique_ptr<std::size_t[], /* free */> offsets; // [n_rows + 1]
    /* aligned buffer */                       pairs;   // [capacity], 64-byte
};

template <class I, class V>
struct csr_slice<layout::tags::aos_tag, I, V> {
    const layout::elementPair<I, V> *pairs;
    std::size_t nnz;

    const V     *binary_search(I col) const;
    template <class Func>
    void         for_each_element(Func &&f) const;
    V            accumulate() const;
};
```

### Build and merge

```cpp
namespace spira::storage {

    template <class LayoutTag, class RowType>
    auto build_csr(const std::vector<RowType> &rows)
        -> csr_storage<LayoutTag,
                       typename RowType::index_type,
                       typename RowType::value_type>;

    template <class LayoutTag, class RowType>
    auto merge_csr(const std::vector<RowType> &rows,
                   csr_storage<LayoutTag, /* ... */> &&old_csr,
                   const std::vector<bool> &dirty)
        -> csr_storage<LayoutTag, /* ... */>;
}
```

`build_csr` is used on the first lock of a matrix. `merge_csr` is used on
every subsequent lock and respects the dirty bitset to skip unchanged rows.

## SIMD kernels

Header: declared inside `src/kernels/` translation units. Namespace
`spira::kernel`. These are global function pointers, set at process load
time by the static initialiser in `src/kernels/dispatch.cpp`.

```cpp
namespace spira::kernel {

    extern double (*sparse_dot_double)(const double   *vals,
                                       const uint32_t *cols,
                                       const double   *x,
                                       std::size_t     n,
                                       std::size_t     x_size);

    extern float (*sparse_dot_float)(const float    *vals,
                                     const uint32_t *cols,
                                     const float    *x,
                                     std::size_t     n,
                                     std::size_t     x_size);
}
```

Both pointers are initialised before `main` runs. You can read them in
your own code if you want to call the kernel directly, but the usual
path is through the SpMV algorithm.

`x_size` is the length of the dense vector; the kernel uses it only for
bounds assertions in debug builds.

### Calling convention

`vals` must point to an array of at least `n` contiguous values of the
matching type. `cols` must point to an array of at least `n` contiguous
column indices. `x` must point to a dense vector of at least `x_size`
values. All column indices must be in the range `[0, x_size)`. The
kernels do not check this outside of debug builds.

There is no alignment requirement on any of the inputs. The kernels use
unaligned loads internally.

## A minimal example

```cpp
#include <spira/spira.hpp>
#include <vector>

int main() {
    using L = spira::layout::tags::soa_tag;
    using spira::matrix;

    matrix<L, uint32_t, double> A(1000, 1000);

    // Open mode: cheap inserts
    A.insert(0, 5, 1.5);
    A.insert(0, 5, 2.0);   // last-write-wins -> stored value is 2.0
    A.insert(1, 0, 3.0);
    A.insert(2, 7, 4.0);

    A.lock();               // build flat CSR

    // Locked mode: fast reads
    std::vector<double> x(1000, 1.0), y(1000);
    spira::serial::algorithms::spmv(A, x, y);

    A.open();
    A.insert(3, 2, 5.0);
    A.lock();               // merge, only row 3 is dirty
}
```
