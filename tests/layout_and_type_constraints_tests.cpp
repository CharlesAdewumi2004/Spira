#include <spira/spira.hpp>
#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>

namespace {

// ---- Helper concepts for testing instantiation/constructibility ----

template <class LayoutTag, class I, class V>
concept MatrixTypeValid = requires { typename spira::matrix<LayoutTag, I, V>; };

template <class LayoutTag, class I, class V>
concept MatrixConstructible = requires {
    spira::matrix<LayoutTag, I, V>{std::size_t{3}, std::size_t{3}};
};

} 


// ------------------ Concept-level tests (static_assert) ------------------

static_assert(spira::concepts::Valueable<double>);
static_assert(spira::concepts::Valueable<float>);

static_assert(spira::concepts::Valueable<int>);

static_assert(spira::concepts::Valueable<std::complex<double>>);

static_assert(!spira::concepts::Valueable<bool>);
static_assert(!spira::concepts::Valueable<std::string>);


static_assert(spira::concepts::Indexable<std::size_t>);
static_assert(spira::concepts::Indexable<std::uint32_t>);
static_assert(spira::concepts::Indexable<std::uint64_t>);

static_assert(!spira::concepts::Indexable<int>);   
static_assert(!spira::concepts::Indexable<bool>);  


// ------------------ Matrix instantiation tests (compile-time) ------------------

static_assert(MatrixTypeValid<spira::layout::tags::aos_tag, std::size_t, double>);
static_assert(MatrixTypeValid<spira::layout::tags::soa_tag, std::uint32_t, float>);
static_assert(MatrixTypeValid<spira::layout::tags::aos_tag, std::uint64_t, int>);
static_assert(MatrixTypeValid<spira::layout::tags::soa_tag, std::size_t, std::complex<double>>);

static_assert(!MatrixTypeValid<spira::layout::tags::aos_tag, int, double>);          
static_assert(!MatrixTypeValid<spira::layout::tags::aos_tag, std::size_t, bool>);  
static_assert(!MatrixTypeValid<spira::layout::tags::aos_tag, std::size_t, std::string>);


// ------------------ Runtime gtest (mostly just smoke tests) ------------------

TEST(LayoutAndTypeConstraintsTest, ConstructAllowedTypes) {
    spira::matrix<spira::layout::tags::aos_tag, std::size_t, double> mat1(5, 5);
    spira::matrix<spira::layout::tags::soa_tag, std::uint32_t, float> mat2(3, 3);
    spira::matrix<spira::layout::tags::aos_tag, std::uint64_t, int> mat3(4, 4);
    spira::matrix<spira::layout::tags::soa_tag, std::size_t, std::complex<double>> mat4(2, 2);

    EXPECT_TRUE(mat1.empty());
    SUCCEED();
}

TEST(LayoutAndTypeConstraintsTest, ConstructibilityConceptChecks) {
    static_assert(MatrixConstructible<spira::layout::tags::aos_tag, std::size_t, double>);
    static_assert(!MatrixConstructible<spira::layout::tags::aos_tag, int, double>);
    static_assert(!MatrixConstructible<spira::layout::tags::aos_tag, std::size_t, bool>);
}
