#include "../kernel_types.h"

__device__ __inline__ bool withinTolerance(float2 a, float2 b, float tsquare) {
    float xdist = a.x - b.x;
    float ydist = a.y - b.y;
    return (xdist * xdist + ydist * ydist) <= tsquare;
}

__device__ dist_t fixedPointDist(complex z, float tsquare, dist_t maxIters, complex p) {
    float2 last = z;
    for(dist_t i = 0; i < maxIters; ++i) {
        z = F(z, p);
        if(withinTolerance(z, last, tsquare))
            return i + 1;
        last = z;
    }
    return -1;
}