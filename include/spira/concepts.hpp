#pragma once

#include "traits.hpp"
#include <concepts>

namespace spira::concepts{
    //used for the col in element
    template<typename I>
    concept Indexable =
        std::unsigned_integral<std::remove_cvref_t<I>> &&
        !traits::is_char_like_v<std::remove_cvref_t<I>>;


    //used for value in element
    template <class V>
    concept Valueable =
        //makes sure we only take real and imaginary numbers
        (!traits::is_char_like_v<V>) &&
        (!std::same_as<std::remove_cv_t<V>, bool>) &&
        // basic semantics
        std::default_initializable<V> &&
        std::copy_constructible<V> &&
        std::movable<V> &&
        std::destructible<V> &&
        std::equality_comparable<V> &&
        // operations (+, +=, *, *=)
        requires (V a, V b) {
            { a + b } -> std::same_as<V>;
            { a * b } -> std::same_as<V>;
            { a += b } -> std::same_as<V&>;
            { a *= b } -> std::same_as<V&>;
        } &&
        //check if the type allows for functions zero (returns whatever zero is considered for the type) and is_zero (checks if some val can be considered zero for a type)
        requires (V x) {
            { traits::ValueTraits<V>::zero() } -> std::same_as<std::remove_cv_t<V>>;
            { traits::ValueTraits<V>::is_zero(x) } -> std::convertible_to<bool>;
        };


}