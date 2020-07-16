#pragma once
#include "safeCall.h"
#include "../Rendering/utils.h"
#include "constants.h"
#include <cuda.h>
#include <cuda_runtime.h>

inline unsigned int ceilDivide(unsigned int x, unsigned int y) {
    return (x + y - 1) / y;
}

template<typename ...Args>
inline void launch_kernel_generic(CUfunction kernel, size_t numThreads, size_t blockSize, Args... args) {
    void* arg_ptrs[] = { &args... };
    unsigned int numBlocks = ceilDivide(numThreads, blockSize);
    CUDA_SAFE(cuLaunchKernel(kernel, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, arg_ptrs, nullptr));
}

inline void launch_kernel_float(CUfunction kernel, float re0, float re1, float im0, float im1, int maxIters, float* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH, float pre, float pim, float metricArg, HostComplex* attractors, size_t numAttractors) {
    launch_kernel_generic(kernel, surfW * surfH, MAIN_KERNEL_BLOCK_SIZE, re0, re1, im0, im1, maxIters, minmaxOut, surface, surfW, surfH, pre, pim, metricArg, attractors, numAttractors);
}

inline void launch_kernel_double(CUfunction kernel, double re0, double re1, double im0, double im1, int maxIters, float* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH, double pre, double pim, double metricArg, HostComplex* attractors, size_t numAttractors) {
    launch_kernel_generic(kernel, surfW * surfH, MAIN_KERNEL_BLOCK_SIZE, re0, re1, im0, im1, maxIters, minmaxOut, surface, surfW, surfH, pre, pim, metricArg, attractors, numAttractors);
}