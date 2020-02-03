#include "kernel.h"
#include <cooperative_groups.h>
#include <cstdio>
#include "math.cuh"

namespace cg = cooperative_groups;

__device__ __inline__ bool withinTolerance(float2 a, float2 b, float tsquare) {
    float xdist = a.x - b.x;
    float ydist = a.y - b.y;
    return (xdist * xdist + ydist * ydist) <= tsquare;
}

__device__ __inline__ float2 getZ(float re0, float re1, float im0, float im1, int width, int height, int x, int y) {
    float reStep = (re1 - re0) / width;
    float imStep = (im1 - im0) / height;
    return make_complex(
        re0 + reStep * x,
        im0 + imStep * y
    );
}

__device__ fpdist_t findFixedPointDist(float2 z, float tsquare, fpdist_t maxIters) {
    float2 last = z;
    for(fpdist_t i = 0; i < maxIters; ++i) {
        z = csin(z);
        if(withinTolerance(z, last, tsquare))
            return i + 1;
        last = z;
    }
    return -1;
}

__device__ __inline__ void debugPrint(const char* str) {
    /*/printf("[kernel %i/%i] %s\n", blockIdx.x, threadIdx.x, str);/**/
}

__global__ void kernel(float re0, float re1, float im0, float im1, float tsquare, fpdist_t maxIters, fpdist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH) {
    __shared__ fpdist2 minmaxBlock[32];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int x = i % surfW;
    int y = i / surfW;
    int area = surfW * surfH;
    if (i - area > 31) return; //The entire warp is excessive

    debugPrint("Start");

    fpdist_t fpDist = -1;
    bool threadIsExcessive = i >= area;
    if (!threadIsExcessive) {
        //Find a z for this thread
        float2 z = getZ(re0, re1, im0, im1, surfW, surfH, x, y);
        fpDist = findFixedPointDist(z, tsquare, maxIters);
    }
    warp.sync();
    debugPrint("FP find done.");

    if (!threadIsExcessive) surf2Dwrite(fpDist, surface, x * sizeof(int), y);
    debugPrint("Surf write done.");

    fpdist_t min_ = (fpDist == -1) ? maxIters + 2 : fpDist;
    fpdist_t max_ = fpDist;
    for (int delta = 16; delta > 0; delta >>= 1) {
        min_ = min(min_, warp.shfl_down(min_, delta));
        max_ = max(max_, warp.shfl_down(max_, delta));
    }
    if (warp.thread_rank() == 0) minmaxBlock[tid / 32] = make_fpdist2(min_, max_);
    block.sync();
    debugPrint("Reduce round 1 done.");

    if (tid < 32) {
        debugPrint("Start reduce round 2.");
        fpdist2 value = minmaxBlock[tid];
        min_ = value.x;
        max_ = value.y;
        for (int delta = 16; delta > 0; delta >>= 1) {
            min_ = min(min_, warp.shfl_down(min_, delta));
            max_ = max(max_, warp.shfl_down(max_, delta));
        }
        if (tid == 0) reinterpret_cast<fpdist2 *>(minmaxOut)[blockIdx.x] = make_fpdist2(min_, max_);
    }
    debugPrint("Done.");
}

void launch_kernel(float re0, float re1, float im0, float im1, float tolerance, fpdist_t maxIters, fpdist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH) {
    kernel<<<surfW * surfH, 1024>>>(re0, re1, im0, im1, tolerance * tolerance, maxIters, minmaxOut, surface, surfW, surfH);
}