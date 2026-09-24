#pragma once

#include <type_traits>
#include <complex>
#include <cmath>
#include <cstdint>
#include <limits>

namespace spira::traits {
    // -------- char-like detection --------
    template <class T>
    struct is_char_like : std::false_type {};
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
    inline constexpr bool is_complex_like_v = is_complex_like<std::remove_cv_t<T>>::value;

    template <class V, class Enable = void>
    struct ValueTraits;

    template <class V>
    struct ValueTraits<V, std::enable_if_t<std::is_arithmetic_v<std::remove_cv_t<V>> || is_complex_like_v<V>>> {
        using value_type = std::remove_cv_t<V>;
        static constexpr value_type zero() noexcept { return value_type{}; }
        static constexpr bool is_zero(const value_type &x) noexcept { return x == value_type{}; }
    };
}
