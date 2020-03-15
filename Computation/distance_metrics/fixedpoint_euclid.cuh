#pragma once
#include "../kernel_types.h"
#include "utils.cuh"

//Euclidean distance from fixed point

__device__ __inline__ dist_t fixedPointDistEuclid(complex z, dist_t maxIters, complex p, float tsquare) {
    const complex original = z;
    complex last = z;
    dist_t i = 0;
    for(; i < maxIters; ++i) {
        z = F(z, p);
        if(withinTolerance(z, last, tsquare))
            break;
        last = z;
    }
    if(i == maxIters)
        return -1;

    return sqrtf(distSquare(original, z));
}