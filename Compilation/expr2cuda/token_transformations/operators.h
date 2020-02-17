#pragma once

#include <map>
#include "../OperatorTraits.h"

const std::map<std::string, OperatorTraits> operatorLookup {
        {"+", {"+", false, 2, "cadd", LEFT_ASSOCIATIVE, 1}},
        {"-", {"-", false, 2, "csub", LEFT_ASSOCIATIVE, 1}},
        {"*", {"*", false, 2, "cmul", LEFT_ASSOCIATIVE, 2}},
        //{"/", {"/", false, 2, "cdiv", LEFT_ASSOCIATIVE, 2}}, todo

        {"or", {"or", true, 2, "cor"}},
        {"xor", {"xor", true, 2, "cxor"}},
        {"and", {"and", true, 2, "cand"}},
        {"not", {"not", true, 1, "cnot"}},

        {"sin", {"sin", true, 1, "csin"}},
        {"cos", {"and", true, 1, "ccos"}},
        {"tan", {"tan", true, 1, "ctan"}},

       // {"abs", {"abs", true, 1, "cabs"}}, todo

        {"neg", {"neg", true, 1, "cneg"}},
        {"conj", {"conj", true, 1, "cconj"}},
};

const std::map<std::string, std::string> unaryOperatorLookup {
        {"+", ""},
        {"-", "neg"},
        {"~", "conj"},
};