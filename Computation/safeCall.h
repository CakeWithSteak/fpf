#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <cuda.h>


#define CUDA_SAFE(x) { cudaErrorHelper((x), __FILE__, __LINE__); }
#define NVRTC_SAFE(x) { nvrtcErrorHelper((x), __FILE__, __LINE__);}

inline void cudaErrorHelper(cudaError_t error, const char* file, int line) {
    if(error != cudaSuccess) {
        std::cerr << "Cuda error " << cudaGetErrorString(error) << " in file \"" << file << "\" at line " << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void cudaErrorHelper(CUresult error, const char* file, int line) {
    if(error != CUDA_SUCCESS) {
        const char* errstr;
        cuGetErrorString(error, &errstr);
        std::cerr << "Cuda error \"" << errstr << "\" in file \"" << file << "\" at line " << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void nvrtcErrorHelper(nvrtcResult error, const char* file, int line) {
    if(error != NVRTC_SUCCESS) {
        std::cerr << "NVRTC error " << nvrtcGetErrorString(error) << " in file \"" << file << "\" at line " << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
