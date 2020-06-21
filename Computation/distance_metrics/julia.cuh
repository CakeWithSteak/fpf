#pragma once
#include "../kernel_types.cuh"

__device__ __inline__ dist_t juliaDist(complex z, int maxIters, complex p, real rsquare, complex c) {
    for(int i = 0; i < maxIters; ++i) {
        if(z.x * z.x + z.y * z.y > rsquare)
            return i;
        z = F(z, p, c);
    }
    return -1;
}