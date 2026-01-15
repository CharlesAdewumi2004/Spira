#pragma once

#include <spira/traits.hpp>
#include <concepts>

namespace spira::concepts
{
    // used for the col in element
    template <typename I>
    concept Indexable =
        std::unsigned_integral<std::remove_cvref_t<I>> &&
        !std::same_as<std::remove_cvref_t<I>, bool> &&
        !traits::is_char_like_v<std::remove_cvref_t<I>> &&
        (sizeof(std::remove_cvref_t<I>) >= 4);

    template <class V>
    concept Valueable =
        (!traits::is_char_like_v<std::remove_cvref_t<V>>) &&
        (!std::same_as<std::remove_cvref_t<V>, bool>) &&
        std::default_initializable<std::remove_cvref_t<V>> &&
        std::copy_constructible<std::remove_cvref_t<V>> &&
        std::movable<std::remove_cvref_t<V>> &&
        std::destructible<std::remove_cvref_t<V>> &&
        std::equality_comparable<std::remove_cvref_t<V>> &&
        requires(std::remove_cvref_t<V> a, std::remove_cvref_t<V> b) {
            { a + b } -> std::same_as<std::remove_cvref_t<V>>;
            { a * b } -> std::same_as<std::remove_cvref_t<V>>;
            { a += b } -> std::same_as<std::remove_cvref_t<V> &>;
            { a *= b } -> std::same_as<std::remove_cvref_t<V> &>;
        } &&
        requires(std::remove_cvref_t<V> x) {
            { traits::ValueTraits<std::remove_cvref_t<V>>::zero() }
            -> std::same_as<std::remove_cvref_t<V>>;
            { traits::ValueTraits<std::remove_cvref_t<V>>::is_zero(x) }
            -> std::convertible_to<bool>;
        };

}