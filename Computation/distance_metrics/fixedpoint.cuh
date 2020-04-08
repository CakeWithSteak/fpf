#pragma once
#include "../kernel_types.h"
#include "../utils.cuh"

__device__ __inline__ dist_t fixedPointDist(complex z, dist_t maxIters, complex p, float tsquare) {
    float2 last = z;
    for(dist_t i = 0; i < maxIters; ++i) {
        z = F(z, p);
        if(withinTolerance(z, last, tsquare))
            return i + 1;
        last = z;
    }
    return -1;
}