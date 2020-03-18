#pragma once
#include <utility>
#include <type_traits>

template <typename T>
class is_ordered {
    template <typename C>
    constexpr static auto test(C c) -> decltype(c < c);

    template <typename C>
    constexpr static std::false_type test(...);
public:
    static constexpr bool value = std::is_same<bool, decltype(test<T>(std::declval<T>()))>::value;
};

template <typename T>
constexpr bool is_ordered_v = is_ordered<T>::value;