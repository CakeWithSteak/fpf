#pragma once
#include <map>
#include "OperatorTraits.h"

const std::map<std::string, OperatorTraits> operatorLookup {
        {"+", {"+", false, 2, "cadd", LEFT_ASSOCIATIVE, 1}},
        {"-", {"-", false, 2, "csub", LEFT_ASSOCIATIVE, 1}},
        {"*", {"*", false, 2, "cmul", LEFT_ASSOCIATIVE, 2}},
        {"/", {"/", false, 2, "cdiv", LEFT_ASSOCIATIVE, 2}},

        {"sin", {"sin", true, 1, "csin"}},
        {"cos", {"and", true, 1, "ccos"}},
        {"tan", {"tan", true, 1, "ctan"}},

        {"abs", {"abs", true, 1, "cabs"}},

        {"neg", {"neg", true, 1, "cneg"}},
        {"conj", {"conj", true, 1, "cconj"}},
        {"Re", {"Re", true, 1, "creal"}},
        {"Im", {"Im", true, 1, "cimag"}},
        {"arg", {"arg", true, 1, "carg"}},

        {"exp", {"exp", true, 1, "cexp"}},
};

//Maps the unary operator to its function form
const std::map<std::string, std::string> unaryOperatorLookup {
        {"+", ""},
        {"-", "neg"},
        {"~", "conj"},
};