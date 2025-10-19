#pragma once

#include <type_traits>
#include <complex>
#include <cmath>
#include <limits>

namespace spira::traits {

    // -------- char-like detection --------
    template <class T> struct is_char_like : std::false_type {};
    template <> struct is_char_like<char> : std::true_type {};
    template <> struct is_char_like<signed char> : std::true_type {};
    template <> struct is_char_like<unsigned char> : std::true_type {};
    #if defined(__cpp_char8_t)
      template <> struct is_char_like<char8_t> : std::true_type {};
    #endif
    template <> struct is_char_like<char16_t> : std::true_type {};
    template <> struct is_char_like<char32_t> : std::true_type {};
    template <> struct is_char_like<wchar_t>  : std::true_type {};

    template <class T>
    inline constexpr bool is_char_like_v = is_char_like<std::remove_cv_t<T>>::value;

    // -------- std::complex detection -----------------------------------
    template <class T> struct is_complex_like : std::false_type {};
    template <class T> struct is_complex_like<std::complex<T>> : std::true_type {};

    template <class T>
    inline constexpr bool is_complex_like_v = is_complex_like<std::remove_cv_t<T>>::value; // FIXED

    // -------- extract underlying type for complex ----------------------
    template <class T> struct complex_value_type { using type = void; };
    template <class T> struct complex_value_type<std::complex<T>> { using type = T; };

    template <class T>
    using complex_value_type_t = typename complex_value_type<std::remove_cv_t<T>>::type;

    // ======================= ValueTraits ================================
    template <class V, class Enable = void>
    struct ValueTraits; // (intentionally incomplete; fallbacks below)

    // ---- arithmetic + complex fallback -------------
    template <class V>
    struct ValueTraits<V, std::enable_if_t<
        std::is_arithmetic_v<std::remove_cv_t<V>> || is_complex_like_v<V>>> {

        using value_type = std::remove_cv_t<V>;

        // Additive identity
        static constexpr value_type zero() noexcept {
            if constexpr (is_complex_like_v<value_type>) {
                using S = complex_value_type_t<value_type>;
                return value_type{S(0), S(0)};
            } else {
                return value_type(0);
            }
        }

        // Epsilon type: integral -> value_type (unused), floating -> same,
        // complex<U> -> U
        using epsilon_type = std::conditional_t<
            std::is_integral_v<value_type>,
            value_type,
            std::conditional_t<
                is_complex_like_v<value_type>,
                complex_value_type_t<value_type>,
                value_type
            >
        >;

        // Consider x "zero"
        static constexpr bool is_zero(const value_type& x,
                                      epsilon_type eps = epsilon_type(0)) noexcept {
            if constexpr (std::is_integral_v<value_type>) {
                (void)eps;
                return x == value_type(0);
            } else if constexpr (std::is_floating_point_v<value_type>) {
                using std::abs;
                return abs(x) <= static_cast<value_type>(eps);
            } else if constexpr (is_complex_like_v<value_type>) {
                using S = complex_value_type_t<value_type>;
                using std::abs;
                return abs(x) <= static_cast<S>(eps);
            } else {
                return x == zero(); // conservative fallback
            }
        }
    };

    // ======================= AccumulationOf =============================
    template <class V, class Enable = void>
    struct AccumulationOf { using type = V; };

    // integral -> promote to 64-bit with signedness preserved
    template <class V>
    struct AccumulationOf<V, std::enable_if_t<std::is_integral_v<std::remove_cv_t<V>>>> {
        using base = std::remove_cv_t<V>;
        using type = std::conditional_t<std::is_signed_v<base>, std::int64_t, std::uint64_t>;
    };

    // float -> double, double->double, long double->long double
    template <class V>
    struct AccumulationOf<V, std::enable_if_t<std::is_floating_point_v<std::remove_cv_t<V>>>> {
        using base = std::remove_cv_t<V>;
        using type = std::conditional_t<std::is_same_v<base, long double>, long double, double>;
    };

    // complex<T> -> complex<AccumulationOf<T>>
    template <class V>
    struct AccumulationOf<V, std::enable_if_t<is_complex_like_v<V>>> {
        using S    = complex_value_type_t<V>;
        using AccS = typename AccumulationOf<S>::type;
        using type = std::complex<AccS>;
    };

    template <class V>
    using AccumulationOf_t = typename AccumulationOf<V>::type;

} // namespace spira::traits
