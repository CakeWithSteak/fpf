#pragma once
#include "../kernel_types.cuh"
#include "../utils.cuh"

//Euclidean distance from fixed point

__device__ __inline__ dist_t fixedPointDistEuclid(complex z, int maxIters, complex p, real tsquare, complex c) {
    const complex original = z;
    complex last = z;
    int i = 0;
    for(; i < maxIters; ++i) {
        z = F(z, p, c);
        if(withinTolerance(z, last, tsquare))
            break;
        last = z;
    }
    if(i == maxIters)
        return -1;

    return sqrtf(distSquare(original, z));
}