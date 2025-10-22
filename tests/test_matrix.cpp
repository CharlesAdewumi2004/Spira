// tests/test_matrix.cpp
#include <gtest/gtest.h>
#include <vector>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <complex>
#include <cstdint>

#include "spira/matrix.hpp"   // your matrix template
#include "spira/element.hpp"  // your element<V,I> with .value and .col
#include "spira/traits.hpp"   // AccumulationOf_t (if split), else remove

// Helper: build an element quickly
template <class V, class I>
static inline spira::element<V, I> E(I col, V val) {
    spira::element<V, I> e{};
    e.col = col;
    e.value = val;
    return e;
}

/* ───────────────────────── Basic construction / empty ───────────────────────── */

TEST(Matrix, Construct_EmptyRows) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 3, C = 5;

    spira::matrix<V, I, R, C> m;
    // All rows exist but start empty
    for (I r = 0; r < R; ++r) {
        for (I c = 0; c < C; ++c) {
            EXPECT_FALSE(m.at(r, c).has_value());
        }
    }
}

/* ───────────────────────── setRow: happy path & invariants ──────────────────── */

TEST(Matrix, SetRow_Valid_SortsAndStores) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 4, C = 8;

    spira::matrix<V, I, R, C> m;

    std::vector<spira::element<V, I>> elems = {
        E<V,I>(5, 50),
        E<V,I>(2, 20),
        E<V,I>(7, 70),
        E<V,I>(3, 30),
    };

    // Should accept and sort by column; row initially empty → true
    EXPECT_TRUE(m.setRow(1, std::move(elems)));

    // Values retrievable in sorted positions
    EXPECT_EQ(m.at(1, 2).value_or(-1), 20);
    EXPECT_EQ(m.at(1, 3).value_or(-1), 30);
    EXPECT_EQ(m.at(1, 5).value_or(-1), 50);
    EXPECT_EQ(m.at(1, 7).value_or(-1), 70);

    // Missing columns → nullopt
    EXPECT_FALSE(m.at(1, 0).has_value());
    EXPECT_FALSE(m.at(1, 4).has_value());
    EXPECT_FALSE(m.at(1, 6).has_value());
}

TEST(Matrix, SetRow_RejectsSecondSetOnSameRow) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 2, C = 5;

    spira::matrix<V, I, R, C> m;

    std::vector<spira::element<V, I>> a = { E<V,I>(1, 10) };
    std::vector<spira::element<V, I>> b = { E<V,I>(2, 20) };

    EXPECT_TRUE(m.setRow(0, std::move(a)));
    // Row already set → policy returns false
    EXPECT_FALSE(m.setRow(0, std::move(b)));
}

TEST(Matrix, SetRow_RowIndexOutOfRangeThrows) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 2, C = 5;

    spira::matrix<V, I, R, C> m;
    std::vector<spira::element<V, I>> elems = { E<V,I>(1, 10) };

    EXPECT_THROW(m.setRow(R, std::move(elems)), std::out_of_range);
}

TEST(Matrix, SetRow_ElementColOutOfRangeThrows) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 2, C = 4;

    spira::matrix<V, I, R, C> m;
    std::vector<spira::element<V, I>> elems = {
        E<V,I>(0, 1),
        E<V,I>(3, 4),
        E<V,I>(4, 5) // invalid: 4 >= C
    };

    EXPECT_THROW(m.setRow(1, std::move(elems)), std::out_of_range);
}

TEST(Matrix, SetRow_DuplicateColumnsThrow) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 3, C = 6;

    spira::matrix<V, I, R, C> m;
    std::vector<spira::element<V, I>> elems = {
        E<V,I>(2, 20),
        E<V,I>(2, 999), // duplicate column
        E<V,I>(5, 50),
    };

    EXPECT_THROW(m.setRow(2, std::move(elems)), std::invalid_argument);
}

TEST(Matrix, SetRow_TooManyElementsThrows) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 1, C = 3;

    spira::matrix<V, I, R, C> m;
    std::vector<spira::element<V, I>> elems = {
        E<V,I>(0, 1),
        E<V,I>(1, 2),
        E<V,I>(2, 3),
        E<V,I>(0, 4) // 4 elements > C (3) → throw
    };

    EXPECT_THROW(m.setRow(0, std::move(elems)), std::out_of_range);
}

/* ───────────────────────────── at(): bounds & behavior ──────────────────────── */

TEST(Matrix, At_OutOfRangeThrows) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 2, C = 3;
    spira::matrix<V, I, R, C> m;

    // out-of-range row
    EXPECT_THROW(m.at(R, 0), std::out_of_range);
    // out-of-range col
    EXPECT_THROW(m.at(0, C), std::out_of_range);
}

TEST(Matrix, At_BoundaryIndexOk) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 1, C = 2;

    spira::matrix<V, I, R, C> m;
    std::vector<spira::element<V, I>> elems = { E<V,I>(C-1, 42) }; // last column
    ASSERT_TRUE(m.setRow(0, std::move(elems)));

    EXPECT_EQ(m.at(0, C-1).value_or(-1), 42);
    EXPECT_FALSE(m.at(0, 0).has_value());
}

TEST(Matrix, At_MissingReturnsNullopt) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 1, C = 4;

    spira::matrix<V, I, R, C> m;
    std::vector<spira::element<V, I>> elems = { E<V,I>(2, 7) };
    ASSERT_TRUE(m.setRow(0, std::move(elems)));

    EXPECT_FALSE(m.at(0, 0).has_value());
    EXPECT_FALSE(m.at(0, 1).has_value());
    EXPECT_TRUE (m.at(0, 2).has_value());
    EXPECT_FALSE(m.at(0, 3).has_value());
}

/* ───────────────────────────── SpMV: correctness ───────────────────────────── */

TEST(Matrix, SpMV_IntBasicAndPromotion) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 3, C = 3;

    spira::matrix<V, I, R, C> m;

    ASSERT_TRUE(m.setRow(0, std::vector{ E<V,I>(0, 1), E<V,I>(2, 2) }));    // [1 0 2]
    ASSERT_TRUE(m.setRow(1, std::vector{ E<V,I>(1, 3) }));                  // [0 3 0]
    ASSERT_TRUE(m.setRow(2, std::vector{ E<V,I>(0, 4), E<V,I>(2, 5) }));    // [4 0 5]

    std::vector<V> x = {10, 20, 30};

    auto y = m.spmv(x);

    // result should be 64-bit accumulation (int -> long long)
    static_assert(std::is_same_v<decltype(y), std::vector<spira::traits::AccumulationOf_t<V>>>, "y type");
    static_assert(std::is_same_v<spira::traits::AccumulationOf_t<int>, std::int64_t>,
              "int accumulates to 64-bit signed");
    static_assert(std::is_same_v<spira::traits::AccumulationOf_t<unsigned>, std::uint64_t>,
                  "unsigned accumulates to 64-bit unsigned");

    ASSERT_EQ(y.size(), R);
    EXPECT_EQ(y[0], 1LL*10 + 2LL*30); // 70
    EXPECT_EQ(y[1], 3LL*20);          // 60
    EXPECT_EQ(y[2], 4LL*10 + 5LL*30); // 190
}

TEST(Matrix, SpMV_FloatPromotesToDouble) {
    using I = std::size_t;
    using V = float;
    constexpr I R = 2, C = 3;

    spira::matrix<V, I, R, C> m;

    ASSERT_TRUE(m.setRow(0, std::vector{ E<V,I>(0, 0.5f), E<V,I>(2, 1.5f) })); // [0.5 0 1.5]
    ASSERT_TRUE(m.setRow(1, std::vector{ E<V,I>(1, 2.0f) }));                  // [0   2 0  ]

    std::vector<V> x = { 2.f, 4.f, 6.f };

    auto y = m.spmv(x);

    // float should accumulate in double
    static_assert(std::is_same_v<spira::traits::AccumulationOf_t<V>, double>, "float -> double");

    ASSERT_EQ(y.size(), R);
    EXPECT_NEAR(y[0], 0.5*2.0 + 1.5*6.0, 1e-12);
    EXPECT_NEAR(y[1], 2.0*4.0,            1e-12);
}

TEST(Matrix, SpMV_ComplexFloatPromotesToComplexDouble) {
    using I = std::size_t;
    using V = std::complex<float>;
    constexpr I R = 2, C = 3;

    spira::matrix<V, I, R, C> m;

    // Row 0: (1+2i) at col 0, (0.5-0.5i) at col 2
    ASSERT_TRUE(m.setRow(0, std::vector{
        E<V,I>(0, V{1.0f, 2.0f}),
        E<V,I>(2, V{0.5f, -0.5f})
    }));
    // Row 1: (3-1i) at col 1
    ASSERT_TRUE(m.setRow(1, std::vector{
        E<V,I>(1, V{3.0f, -1.0f})
    }));

    std::vector<V> x = {
        V{2.0f, 0.0f},  // col 0
        V{0.0f, 1.0f},  // col 1
        V{1.0f, 2.0f}   // col 2
    };

    auto y = m.spmv(x);

    using Acc = spira::traits::AccumulationOf_t<V>;
    static_assert(std::is_same_v<Acc, std::complex<double>>, "complex<float> -> complex<double>");

    ASSERT_EQ(y.size(), R);

    // Expected:
    // y0 = (1+2i)*(2+0i) + (0.5-0.5i)*(1+2i)
    //    = (2+4i) + (0.5 + 1.0i - 0.5i - 1.0i^2) = (2+4i) + (1.5 + 0.5i) = (3.5 + 4.5i)
    // y1 = (3-1i)*(0+1i) = (3i - i^2) = (1 + 3i)
    EXPECT_NEAR(y[0].real(), 3.5, 1e-12);
    EXPECT_NEAR(y[0].imag(), 4.5, 1e-12);
    EXPECT_NEAR(y[1].real(), 1.0, 1e-12);
    EXPECT_NEAR(y[1].imag(), 3.0, 1e-12);
}

/* ─────────────────────────── Size / index sanity ───────────────────────────── */

TEST(Matrix, IndexCastsSafeForSmallExamples) {
    // smoke test that we can construct and multiply with a wider I than size_t
    // (in practice your ctor should guard sizes vs SIZE_MAX; here we keep sizes tiny)
    using I = unsigned long long; // possibly wider than size_t on 32-bit
    using V = int;
    constexpr I R = 2, C = 2;

    spira::matrix<V, I, R, C> m;
    ASSERT_TRUE(m.setRow(0, std::vector{ E<V,I>(0, 1), E<V,I>(1, 2) }));
    ASSERT_TRUE(m.setRow(1, std::vector{ E<V,I>(1, 3) }));

    std::vector<V> x = { 10, 20 };
    auto y = m.spmv(x);

    ASSERT_EQ(y.size(), static_cast<std::size_t>(R));
    EXPECT_EQ(y[0], 1LL*10 + 2LL*20);
    EXPECT_EQ(y[1], 3LL*20);
}
