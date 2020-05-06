#pragma once
#include "../kernel_types.h"

__device__ __inline__ dist_t juliaDist(complex z, dist_t maxIters, complex p, float rsquare, complex c) {
    for(dist_t i = 0; i < maxIters; ++i) {
        if(z.x * z.x + z.y * z.y > rsquare)
            return i;
        z = F(z, p, c);
    }
    return -1;
}