#pragma once
#include <string_view>
#include <cuda_runtime.h>
#include <cuda.h>
#include <utility>
#include <vector>
#include <filesystem>
#include "../modes.h"

class NvrtcCompiler {
    CUmodule module = nullptr;
    std::filesystem::path cudaIncludePath;

    std::vector<char*> getCompileArgs(const ModeInfo& mode, bool doublePrec);
    static std::string getSM();
    static std::string defineMacro(const std::string& name);
public:
    explicit NvrtcCompiler(std::filesystem::path cudaIncludePath) : cudaIncludePath{std::move(cudaIncludePath)} {}
    //Compiles C++ code to PTX, loads it intro a new module, and returns the address of the compiled function
    std::vector<CUfunction>
    Compile(std::string_view code, std::string_view filename, std::vector<std::string_view> functionNames, const ModeInfo& mode, bool doublePrec);
    void Unload();
    ~NvrtcCompiler();
};


