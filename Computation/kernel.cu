/*
 * This file is preprocessed to create kernel.ii, which is included in runtime_template.h
 * Normal preprocessor directives are processed build-time, while the ones marked with DEFER_TO_NVRTC_PREPROCESSOR
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
#include "metrics.cuh"
DEFER_TO_NVRTC_PREPROCESSOR #include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __inline__ float2 getZ(float re0, float re1, float im0, float im1, int width, int height, int x, int y) {
    float reStep = (re1 - re0) / width;
    float imStep = (im1 - im0) / height;
    return make_complex(
        re0 + reStep * x,
        im0 + imStep * y
    );
}

__global__ void kernel(float re0, float re1, float im0, float im1, float tsquare, dist_t maxIters, dist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH, float pre, float pim) {
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
        fpDist = DIST_F(z, tsquare, maxIters, make_complex(pre, pim));
    }
    warp.sync();

    if (!threadIsExcessive) surf2Dwrite(fpDist, surface, x * sizeof(int), y);

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
        if (tid == 0) reinterpret_cast<dist2 *>(minmaxOut)[blockIdx.x] = make_dist2(min_, max_);
    }
}

__device__ __inline__ complex F(complex z, complex p) {
    /*Generated code goes here*/
//}
#ifndef BUILD_FOR_NVRTC
#pragma clang diagnostic pop
#endif