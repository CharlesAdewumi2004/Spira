#include <gtest/gtest.h>

#include <spira/spira.hpp>
#include "spira/element.hpp"
#include "spira/traits.hpp"
#include "spira/concepts.hpp"


#include <type_traits>
#include <string>
#include <complex>
#include <cstdint>

// Namespace aliases for brevity
namespace ct = spira::traits;
namespace cc = spira::concepts;
namespace se = spira;

// ---------------------------------------------
// Helper: detect whether element<V,I> is well-formed
// (so we can negative-test without breaking the build)
// ---------------------------------------------
template<typename V, typename I>
concept ElementWellFormed = requires {
    typename se::element<V, I>;
};

// ---------------------------------------------
// Concept sanity checks (compile-time)
// ---------------------------------------------

// ---- Indexable ----
// Valid
static_assert(cc::Indexable<std::size_t>);
static_assert(cc::Indexable<unsigned int>);
static_assert(cc::Indexable<std::uint16_t>);
static_assert(cc::Indexable<std::uint64_t>);

// Invalid
static_assert(!cc::Indexable<int>);             // signed -> not allowed
static_assert(!cc::Indexable<long long>);       // signed
static_assert(!cc::Indexable<char>);            // char-like
static_assert(!cc::Indexable<wchar_t>);         // char-like
static_assert(!cc::Indexable<signed char>);     // char-like
static_assert(!cc::Indexable<unsigned char>);   // char-like
static_assert(!cc::Indexable<bool>);            //bool

// ---- Valueable ----
// Valid reals
static_assert(cc::Valueable<int>);
static_assert(cc::Valueable<unsigned>);
static_assert(cc::Valueable<float>);
static_assert(cc::Valueable<double>);
static_assert(cc::Valueable<long double>);

// Valid complex
static_assert(cc::Valueable<std::complex<float>>);
static_assert(cc::Valueable<std::complex<double>>);

// Invalid value types
static_assert(!cc::Valueable<char>);            // char-like
static_assert(!cc::Valueable<wchar_t>);
static_assert(!cc::Valueable<bool>);            // explicitly excluded
static_assert(!cc::Valueable<std::string>);     // fails ops/traits requirements

// ---------------------------------------------
// ElementWellFormed
// ---------------------------------------------

// Valid pairs
static_assert(ElementWellFormed<int, std::size_t>);
static_assert(ElementWellFormed<double, std::uint32_t>);
static_assert(ElementWellFormed<std::complex<float>, std::uint16_t>);
static_assert(ElementWellFormed<std::complex<double>, std::size_t>);

// Invalid pairs
static_assert(!ElementWellFormed<int, int>);                    // signed index
static_assert(!ElementWellFormed<int, char>);                   // char-like index
static_assert(!ElementWellFormed<char, std::size_t>);           // char value
static_assert(!ElementWellFormed<std::string, std::size_t>);    // non-Valueable value
static_assert(!ElementWellFormed<bool, std::size_t>);           // bool not Valueable (by your concept)

// ---------------------------------------------
// Runtime tests for storage/semantics
// ---------------------------------------------

TEST(ElementTest, ConstructStoresValueAndCol_IntIndex) {
    se::element<int, std::size_t> e{42, 7};
    EXPECT_EQ(e.value, 42);
    EXPECT_EQ(e.col, 7u);
}

TEST(ElementTest, ConstructStoresValueAndCol_DoubleIndex32) {
    se::element<double, std::uint32_t> e{3.14, 12};
    EXPECT_DOUBLE_EQ(e.value, 3.14);
    EXPECT_EQ(e.col, 12u);
}

TEST(ElementTest, ConstructStoresValueAndCol_ComplexIndex16) {
    se::element<std::complex<float>, std::uint16_t> e{{2.0f, -5.0f}, 9};
    EXPECT_FLOAT_EQ(e.value.real(), 2.0f);
    EXPECT_FLOAT_EQ(e.value.imag(), -5.0f);
    EXPECT_EQ(e.col, static_cast<std::uint16_t>(9));
}

// ---------------------------------------------
// ValueTraits runtime behavior
// ---------------------------------------------

TEST(ValueTraitsTest, ZeroForScalars) {
    using VT = ct::ValueTraits<double>;
    auto z = VT::zero();
    EXPECT_DOUBLE_EQ(z, 0.0);
    EXPECT_TRUE(VT::is_zero(0.0));
    EXPECT_FALSE(VT::is_zero(1.0));
}

TEST(ValueTraitsTest, ZeroForComplex) {
    using VT = ct::ValueTraits<std::complex<double>>;
    auto z = VT::zero();
    EXPECT_DOUBLE_EQ(z.real(), 0.0);
    EXPECT_DOUBLE_EQ(z.imag(), 0.0);
    EXPECT_TRUE(VT::is_zero(std::complex<double>(0.0, 0.0)));
    EXPECT_FALSE(VT::is_zero(std::complex<double>(1e-3, 0.0))); // with default eps=0, non-zero
}

TEST(ValueTraitsTest, IsZeroWithEpsilonFloating) {
    using VT = ct::ValueTraits<float>;
    // With eps = 1e-4 consider tiny value as zero
    EXPECT_TRUE(VT::is_zero(5e-5f, 1e-4f));
    EXPECT_FALSE(VT::is_zero(2e-4f, 1e-4f));
}

TEST(ValueTraitsTest, IsZeroWithEpsilonComplex) {
    using VT = ct::ValueTraits<std::complex<float>>;
    EXPECT_TRUE(VT::is_zero(std::complex<float>(5e-5f, 0.0f), 1e-4f));
    EXPECT_FALSE(VT::is_zero(std::complex<float>(2e-4f, 0.0f), 1e-4f));
}

// ---------------------------------------------
// AccumulationOf mapping
// ---------------------------------------------

TEST(AccumulationOfTest, IntegralPromotesPreservingSignedness) {
    using S1 = ct::AccumulationOf_t<int>;
    using S2 = ct::AccumulationOf_t<unsigned>;
    static_assert(std::is_same_v<S1, std::int64_t>);
    static_assert(std::is_same_v<S2, std::uint64_t>);
}

TEST(AccumulationOfTest, FloatingPromotesToDoubleExceptLongDouble) {
    using A = ct::AccumulationOf_t<float>;
    using B = ct::AccumulationOf_t<double>;
    using C = ct::AccumulationOf_t<long double>;
    static_assert(std::is_same_v<A, double>);
    static_assert(std::is_same_v<B, double>);
    static_assert(std::is_same_v<C, long double>);
}

TEST(AccumulationOfTest, ComplexPromotesInner) {
    using A = ct::AccumulationOf_t<std::complex<float>>;
    using B = ct::AccumulationOf_t<std::complex<double>>;
    static_assert(std::is_same_v<A, std::complex<double>>);
    static_assert(std::is_same_v<B, std::complex<double>>);
}
