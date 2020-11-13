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

std::string getSM() {
    cudaDeviceProp props;
    CUDA_SAFE(cudaGetDeviceProperties(&props, 0));
    return "compute_" + std::to_string(props.major) + std::to_string(props.minor);
}

//todo read env var or add a cli switch to change cuda install dir
std::string getIncludePath() {
#if defined(unix) || defined(__unix) || defined(__unix__)
    return "/usr/local/cuda/include/";
#elif defined(_MSC_VER) || defined(__WIN32)
    return "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include"; //todo avoid hardcoding a specific version
#else
#error "Unsupported operating system."
#endif
}

std::vector<char*> getCompileArgs(const ModeInfo& mode, bool doublePrec) {
    std::vector<std::string> args;
    args.emplace_back("--gpu-architecture=" + getSM());
    args.emplace_back("--include-path=" + getIncludePath());
    args.emplace_back("-std=c++14");
    args.emplace_back("-ewp");
    args.push_back(defineMacro(mode.metricInternalName));
    if(mode.capturing)
        args.push_back(defineMacro("CAPTURING"));
    if(mode.staticMinMax.has_value() || mode.isAttractor)
        args.push_back(defineMacro("NO_MINMAX"));
    if(doublePrec)
        args.push_back(defineMacro("PREC_DOUBLE"));
    else
        args.push_back(defineMacro("PREC_FLOAT"));

    std::vector<char*> res;
    for(int i = 0; i < args.size(); ++i) {
        res.push_back(new char[args[i].size() + 1]);
        std::copy(args[i].data(), args[i].data() + args[i].size() + 1, res[i]);
    }
    return res;
}

std::vector<CUfunction>
NvrtcCompiler::Compile(std::string_view code, std::string_view filename, std::vector<std::string_view> functionNames, const ModeInfo& mode, bool doublePrec) {
    nvrtcProgram program;
    NVRTC_SAFE(nvrtcCreateProgram(&program, code.data(), filename.data(), 0, nullptr, nullptr));

    for(auto name : functionNames)
        NVRTC_SAFE(nvrtcAddNameExpression(program, name.data()));

    const auto compileArgs = getCompileArgs(mode, doublePrec);
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