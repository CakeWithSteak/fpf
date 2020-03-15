#pragma once
#include <string_view>
#include <cuda_runtime.h>
#include <cuda.h>
#include "../Computation/kernel_types.h"

class NvrtcCompiler {
    CUmodule module = nullptr;
public:
    //Compiles C++ code to PTX, loads it intro a new module, and returns the address of the compiled function
    CUfunction Compile(std::string_view code, std::string_view filename, std::string_view functionName, DistanceMetric metric);
    void Unload();
    ~NvrtcCompiler();
};


