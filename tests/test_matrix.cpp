// tests/test_matrix.cpp
#include <gtest/gtest.h>
#include <vector>
#include <optional>
#include <stdexcept>

#include "spira/matrix.hpp"   // your matrix template
#include "spira/element.hpp"  // your element<V,I> with .value and .col

// Helper: build an element quickly
template <class V, class I>
static inline spira::element<V, I> E(I col, V val) {
    spira::element<V, I> e{};
    e.col = col;
    e.value = val;
    return e;
}

TEST(Matrix, Construct_EmptyRows) {
    using I = std::size_t;
    using V = int;
    constexpr I R = 3, C = 5;

    spira::matrix<V, I, R, C> m;
    // All rows exist but start empty
    for (I r = 0; r < R; ++r) {
        // at() on empty row should be nullopt for any valid col
        for (I c = 0; c < C; ++c) {
            EXPECT_FALSE(m.at(r, c).has_value());
        }
    }
}

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

    // Values retrievable
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

