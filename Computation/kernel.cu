/*
 * This file is preprocessed to create kernel.ii, which is included in runtime_template.h
 * Normal preprocessor directives are processed build-time, while the ones marked with RUNTIME
 *  are processed runtime by NVRTC.
 */

// Hack for proper code insights
#ifndef BUILD_FOR_NVRTC
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <surface_types.h>
#endif

#include "kernel_macros.cuh"
#include "kernel_types.h"
#include "math.cuh"
#include "metrics.h"
#include "utils.cuh"
#include "attractor.cuh"
RUNTIME #include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __inline__ complex pickStartingZ(const complex& z) {
RUNTIME #ifdef CAPTURING
    return make_complex(0,0);
RUNTIME #else
    return z;
RUNTIME #endif
}

__device__ __inline__ dist_t callDistanceFunction(complex z, dist_t arg1, complex p, float arg2, const complex* attractors, size_t numAttractors) {
RUNTIME #ifdef ATTRACTOR
    return whichAttractor(pickStartingZ(z), arg1, p, arg2, z, attractors, numAttractors);
RUNTIME #else
    return DIST_F(pickStartingZ(z), arg1, p, arg2, z);
RUNTIME #endif
}

__global__ void kernel(float re0, float re1, float im0, float im1, dist_t maxIters, dist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH, float pre, float pim, float metricArg, const complex* attractors, size_t numAttractors) {
    __shared__ dist2 minmaxBlock[32];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int x = i % surfW;
    int y = i / surfW;
    int area = surfW * surfH;
    if (i - area > 31) return; //The entire warp is excessive

    dist_t fpDist = -1;
    bool threadIsExcessive = i >= area;
    if (!threadIsExcessive) {
        //Find a z for this thread
        float2 z = getZ(re0, re1, im0, im1, surfW, surfH, x, y);
        fpDist = callDistanceFunction(z, maxIters, make_complex(pre, pim), metricArg, attractors, numAttractors);
    }
    warp.sync();

    if (!threadIsExcessive) surf2Dwrite(fpDist, surface, x * sizeof(int), y);

RUNTIME #ifndef NO_MINMAX

    //Find the min/max of this warp and write it to minmaxBlock
    dist_t min_ = (fpDist == -1) ? maxIters + 2 : fpDist;
    dist_t max_ = fpDist;
    for (int delta = 16; delta > 0; delta >>= 1) {
        min_ = min(min_, warp.shfl_down(min_, delta));
        max_ = max(max_, warp.shfl_down(max_, delta));
    }
    if (warp.thread_rank() == 0) minmaxBlock[tid / 32] = make_dist2(min_, max_);
    block.sync();

    //The first warp calculates the min/max of the whole block and writes it to the output buffer
    if (tid < 32) {
        dist2 value = minmaxBlock[tid];
        min_ = value.x;
        max_ = value.y;
        for (int delta = 16; delta > 0; delta >>= 1) {
            min_ = min(min_, warp.shfl_down(min_, delta));
            max_ = max(max_, warp.shfl_down(max_, delta));
        }
        if (tid == 0) reinterpret_cast<dist2*>(minmaxOut)[blockIdx.x] = make_dist2(min_, max_);
    }

RUNTIME #endif
}

__global__ void genFixedPointPath(float re, float im, int maxSteps, float tsquare, complex* output, int* outputLength, float pre, float pim) {
    complex c = make_complex(re, im);
RUNTIME #ifdef CAPTURING
    complex z = make_complex(0,0);
    output[0] = c;
    output[1] = z;
    int i = 2;
RUNTIME #else
    complex z = c;
    output[0] = z;
    int i = 1;
RUNTIME #endif

    complex p = make_complex(pre, pim);
    complex last = z;
    for(; i < maxSteps; ++i) {
        z = F(z, p, c);
        output[i] = z;
        if(withinTolerance(z, last, tsquare))
            break;
        last = z;
    }
    *outputLength = min(i + 1, maxSteps);
}

#include "transform_line.cuh"

__device__ __inline__ complex F(complex z, complex p, complex c) {
    /*Generated code goes here*/
//}
#ifndef BUILD_FOR_NVRTC
#pragma clang diagnostic pop
#endif