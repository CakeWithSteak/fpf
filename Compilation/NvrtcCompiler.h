#pragma once
#include <string_view>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include "../modes.h"

class NvrtcCompiler {
    CUmodule module = nullptr;
public:
    //Compiles C++ code to PTX, loads it intro a new module, and returns the address of the compiled function
    std::vector<CUfunction>
    Compile(std::string_view code, std::string_view filename, std::vector<std::string_view> functionNames, const ModeInfo& mode, bool doublePrec);
    void Unload();
    ~NvrtcCompiler();
};


