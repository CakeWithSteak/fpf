#pragma once
#include "../kernel_types.cuh"
#include "../utils.cuh"

__device__ __inline__ dist_t fixedPointDist(complex z, int maxIters, complex p, real tsquare, complex c) {
    complex last = z;
    for(int i = 0; i < maxIters; ++i) {
        z = F(z, p, c);
        if(withinTolerance(z, last, tsquare))
            return i + 1;
        last = z;
    }
    return -1;
}