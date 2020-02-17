#pragma once
#include <map>
#include <cassert>
#include "token_transformations/operators.h"

template <typename K, typename V>
inline bool mapContains(const std::map<K,V>& m, const K& key) {
    return m.find(key) != m.cend();
}

inline OperatorTraits getTraits(const Token& t) {
    assert(t.type == TokenType::OPERATOR);
    return operatorLookup.at(t.value);
}