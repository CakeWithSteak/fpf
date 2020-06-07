#include "NvrtcCompiler.h"
#include <nvrtc.h>
#include <vector>
#include <string_view>
#include <memory>
#include "../Computation/safeCall.h"
#include "../modes.h"

std::string defineMacro(const std::string& name) {
    std::string prefix = "--define-macro=";
    return prefix + name;
}

std::vector<char*> getCompileArgs(const ModeInfo& mode) {
    std::vector<std::string> args;
    args.push_back("--gpu-architecture=compute_61"); //todo autodetect sm
    args.push_back("--include-path=/usr/local/cuda/include/");
    args.push_back("-std=c++14");
    args.push_back("-ewp");
    args.push_back(defineMacro(mode.metricInternalName));
    if(mode.capturing)
        args.push_back(defineMacro("CAPTURING"));
    if(mode.staticMinMax.has_value() || mode.isAttractor)
        args.push_back(defineMacro("NO_MINMAX"));

    std::vector<char*> res;
    for(int i = 0; i < args.size(); ++i) {
        res.push_back(new char[args[i].size() + 1]);
        std::copy(args[i].data(), args[i].data() + args[i].size() + 1, res[i]);
    }
    return res;
}

std::vector<CUfunction>
NvrtcCompiler::Compile(std::string_view code, std::string_view filename, std::vector<std::string_view> functionNames, const ModeInfo& mode) {
    nvrtcProgram program;
    NVRTC_SAFE(nvrtcCreateProgram(&program, code.data(), filename.data(), 0, nullptr, nullptr));

    for(auto name : functionNames)
        NVRTC_SAFE(nvrtcAddNameExpression(program, name.data()));

    const auto compileArgs = getCompileArgs(mode);
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
    NVRTC_SAFE(nvrtcGetPTX(program, ptx.get()));

    std::vector<const char*> loweredNames(functionNames.size()); //The mangled names of the functions
    for(int i = 0; i < functionNames.size(); ++i)
        NVRTC_SAFE(nvrtcGetLoweredName(program, functionNames[i].data(), &loweredNames[i]));

    CUDA_SAFE(cuModuleLoadDataEx(&module, ptx.get(), 0, nullptr, nullptr));

    std::vector<CUfunction> functions(functionNames.size());
    for(int i = 0; i < functions.size(); ++i)
        CUDA_SAFE(cuModuleGetFunction(&functions[i], module, loweredNames[i]));
    NVRTC_SAFE(nvrtcDestroyProgram(&program));
    for(char* arg : compileArgs)
        delete[] arg;
    return functions;
}

void NvrtcCompiler::Unload() {
    if(module)
        CUDA_SAFE(cuModuleUnload(module));
}

NvrtcCompiler::~NvrtcCompiler() { Unload(); }