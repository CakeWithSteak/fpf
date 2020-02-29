#pragma once
#include <string>

//kernel.ii is generated build-time
const std::string runtimeTemplateCode {
    #include "kernel.ii"
};