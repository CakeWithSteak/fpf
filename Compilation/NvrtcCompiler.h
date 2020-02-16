#pragma once
#include <string_view>
#include <cuda_runtime.h>
#include <cuda.h>

class NvrtcCompiler {
    CUmodule module = nullptr;
public:
    CUfunction Compile(std::string_view code, std::string_view filename, std::string_view functionName);
    void Unload();
    ~NvrtcCompiler();
};


