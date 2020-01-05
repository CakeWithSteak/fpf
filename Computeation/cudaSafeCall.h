#pragma once
#include <iostream>
#include <cuda_runtime_api.h>

#define CUDA_SAFE(x) { errorHelper((x), __FILE__, __LINE__); }
inline void errorHelper(cudaError_t error, const char* file, int line) {
    if(error != cudaSuccess) {
        std::cerr << "Cuda error " << cudaGetErrorString(error) << " in file \"" << file << "\" at line " << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
