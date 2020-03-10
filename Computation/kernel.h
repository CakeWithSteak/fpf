#pragma once
#include "kernel_types.h"
#include "safeCall.h"
#include <cuda.h>
#include <cuda_runtime.h>

inline void launch_kernel(CUfunction kernel, float re0, float re1, float im0, float im1, float tolerance, fpdist_t maxIters, fpdist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH, float pre, float pim) {
    constexpr unsigned int BLOCK_SIZE = 1024;

    auto tsquare = tolerance * tolerance;
    void* args[] = {&re0, &re1, &im0, &im1, &tsquare, &maxIters, &minmaxOut, &surface, &surfW, &surfH, &pre, &pim};
    unsigned int numBlocks = (surfW * surfH) / BLOCK_SIZE;
    if((surfW * surfH) % BLOCK_SIZE != 0)
        ++numBlocks;
    CUDA_SAFE(cuLaunchKernel(kernel, numBlocks, 1, 1, BLOCK_SIZE, 1, 1, 0, nullptr, args, nullptr));
}