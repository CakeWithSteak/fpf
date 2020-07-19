#pragma once
#include "../kernel_types.cuh"

// Exactly the same as julia, but always runs all maxIters iterations,
// thus correctly coloring orbits which leave the escape radius only to go back inside later
__device__ __inline__ dist_t juliaCompleteDist(complex z, int maxIters, complex p, real rsquare, complex c) {
    int leftAfter = -1;
    for(int i = 0; i < maxIters; ++i) {
        if(z.x * z.x + z.y * z.y > rsquare)
            leftAfter = i;
        z = F(z, p, c);
    }
    if(z.x * z.x + z.y * z.y <= rsquare)
        return -1;
    return leftAfter;
}