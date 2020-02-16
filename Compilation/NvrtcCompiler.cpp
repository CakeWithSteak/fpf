#include "NvrtcCompiler.h"
#include <nvrtc.h>
#include "../Computation/safeCall.h"

CUfunction NvrtcCompiler::Compile(std::string_view code, std::string_view filename, std::string_view functionName) {
    nvrtcProgram program;
    NVRTC_SAFE(nvrtcCreateProgram(&program, code.data(), filename.data(), 0, nullptr, nullptr));
    NVRTC_SAFE(nvrtcAddNameExpression(program, functionName.data()));
    const char* compileArgs[] = {"--gpu-architecture=compute_61", "--include-path=/usr/local/cuda/include/", "-std=c++14"};
    nvrtcResult compileResult = nvrtcCompileProgram(program, 3, compileArgs); //todo more options for optimization
    if(compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        NVRTC_SAFE(nvrtcGetProgramLogSize(program, &logSize));
        std::string log(logSize, 0);
        NVRTC_SAFE(nvrtcGetProgramLog(program, log.data()));
        throw std::runtime_error(log);
    }
    size_t ptxSize;
    NVRTC_SAFE(nvrtcGetPTXSize(program, &ptxSize));
    char* ptx = new char[ptxSize];
    NVRTC_SAFE(nvrtcGetPTX(program, ptx));
    const char* loweredName;
    NVRTC_SAFE(nvrtcGetLoweredName(program, functionName.data(), &loweredName));

    CUDA_SAFE(cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr));
    delete[] ptx;

    CUfunction function;
    CUDA_SAFE(cuModuleGetFunction(&function, module, loweredName));
    NVRTC_SAFE(nvrtcDestroyProgram(&program));
    return function;
}

void NvrtcCompiler::Unload() {
    if(module)
        CUDA_SAFE(cuModuleUnload(module));
}

NvrtcCompiler::~NvrtcCompiler() { Unload(); }