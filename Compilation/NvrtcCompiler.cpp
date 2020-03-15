#include "NvrtcCompiler.h"
#include <nvrtc.h>
#include <vector>
#include <string_view>
#include <memory>
#include "../Computation/safeCall.h"

std::string defineMacro(const std::string& name) {
    std::string prefix = "--define-macro=";
    return prefix + name;
}

std::vector<char*> getCompileArgs(DistanceMetric metric) {
    std::vector<std::string> args;
    args.push_back("--gpu-architecture=compute_61"); //todo autodetect sm
    args.push_back("--include-path=/usr/local/cuda/include/");
    args.push_back("-std=c++14");
    args.push_back("-ewp");
    args.push_back(defineMacro(metricMacroMap.at(metric)));
    std::vector<char*> res;
    for(int i = 0; i < args.size(); ++i) {
        res.push_back(new char[args[i].size() + 1]);
        std::copy(args[i].data(), args[i].data() + args[i].size() + 1, res[i]);
    }
    return res;
}

CUfunction NvrtcCompiler::Compile(std::string_view code, std::string_view filename, std::string_view functionName, DistanceMetric metric) {
    nvrtcProgram program;
    NVRTC_SAFE(nvrtcCreateProgram(&program, code.data(), filename.data(), 0, nullptr, nullptr));
    NVRTC_SAFE(nvrtcAddNameExpression(program, functionName.data()));
    const auto compileArgs = getCompileArgs(metric);
    nvrtcResult compileResult = nvrtcCompileProgram(program, compileArgs.size(), compileArgs.data());
    if(compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        NVRTC_SAFE(nvrtcGetProgramLogSize(program, &logSize));
        std::string log(logSize, 0);
        NVRTC_SAFE(nvrtcGetProgramLog(program, log.data()));
        throw std::runtime_error(log);
    }
    size_t ptxSize;
    NVRTC_SAFE(nvrtcGetPTXSize(program, &ptxSize));
    auto ptx = std::make_unique<char[]>(ptxSize);
    //char* ptx = new char[ptxSize];
    NVRTC_SAFE(nvrtcGetPTX(program, ptx.get()));
    const char* loweredName; //The mangled name of the function
    NVRTC_SAFE(nvrtcGetLoweredName(program, functionName.data(), &loweredName));

    CUDA_SAFE(cuModuleLoadDataEx(&module, ptx.get(), 0, nullptr, nullptr));

    CUfunction function;
    CUDA_SAFE(cuModuleGetFunction(&function, module, loweredName));
    NVRTC_SAFE(nvrtcDestroyProgram(&program));
    for(char* arg : compileArgs)
        delete[] arg;
    return function;
}

void NvrtcCompiler::Unload() {
    if(module)
        CUDA_SAFE(cuModuleUnload(module));
}

NvrtcCompiler::~NvrtcCompiler() { Unload(); }