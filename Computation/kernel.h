#pragma once
#include "kernel_types.h"
#include "safeCall.h"
#include <cuda.h>
#include <cuda_runtime.h>

inline void launch_kernel(CUfunction kernel, float re0, float re1, float im0, float im1, float tolerance, fpdist_t maxIters, fpdist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH, float pre, float pim) {
    auto tsquare = tolerance * tolerance;
    void* args[] = {&re0, &re1, &im0, &im1, &tsquare, &maxIters, &minmaxOut, &surface, &surfW, &surfH, &pre, &pim};
    CUDA_SAFE(cuLaunchKernel(kernel, surfW * surfH, 1, 1, 1024, 1, 1, 0, nullptr, args, nullptr));
}