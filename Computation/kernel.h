#pragma once
#include "kernel_types.h"
#include "safeCall.h"
#include <cuda.h>
#include <cuda_runtime.h>

inline unsigned int ceilDivide(unsigned int x, unsigned int y) {
    return (x + y - 1) / y;
}

inline void launch_kernel(CUfunction kernel, float re0, float re1, float im0, float im1, dist_t maxIters, dist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH, float pre, float pim, float metricArg) {
    constexpr unsigned int BLOCK_SIZE = 1024;

    void* args[] = {&re0, &re1, &im0, &im1, &maxIters, &minmaxOut, &surface, &surfW, &surfH, &pre, &pim, &metricArg};
    unsigned int numBlocks = ceilDivide(surfW * surfH, BLOCK_SIZE);
    CUDA_SAFE(cuLaunchKernel(kernel, numBlocks, 1, 1, BLOCK_SIZE, 1, 1, 0, nullptr, args, nullptr));
}