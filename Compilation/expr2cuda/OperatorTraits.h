#pragma once
#include <string>

enum Associativity {
    NOT_ASSOCIATIVE,
    RIGHT_ASSOCIATIVE,
    LEFT_ASSOCIATIVE
};

struct OperatorTraits {
    std::string name;
    bool isFunction;
    int arity;
    Associativity associativity; // Ignored for functions
    int precedence; // Ignored for functions
    std::string cudaName;

    OperatorTraits(const std::string& name, bool isFunction, int arity, const std::string& cudaName, Associativity associativity = NOT_ASSOCIATIVE, int precedence = 0) :
        name{name}, isFunction{isFunction}, associativity{associativity}, precedence{precedence}, cudaName{cudaName}, arity{arity} {}
};
