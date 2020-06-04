#include "kernel_types.h"
#include "utils.cuh"

#define NANF (0.0f / 0.0f)

__global__ void findAttractors(float re0, float re1, float im0, float im1, dist_t maxIters, float pre, float pim, float tsquare, int width, int height, complex* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;;
    int x = tid % width;
    int y = tid / width;
    int area = width * height;
    if (tid >= area) return;

    complex z = getZ(re0, re1, im0, im1, width, height, x, y);
    complex p = make_complex(pre, pim);

RUNTIME #ifdef CAPTURING
    complex c = z;
RUNTIME #else
    complex c = make_complex(0,0);
RUNTIME #endif

    complex last = z;
    for(int i = 0; i < maxIters; ++i) {
        z = F(z, p, c);
        if(distSquare(z, last) <= tsquare) {
            output[tid] = z;
            return;
        }
        last = z;
    }
    output[tid] = make_complex(NANF, NANF);
}

__device__ __inline__ dist_t whichAttractor(complex z, dist_t maxIters, complex p, float tsquare, complex c, const complex* attractors, size_t numAttractors) {
    float2 last = z;
    dist_t i = 0;
    for(; i < maxIters; ++i) {
        z = F(z, p, c);
        if(withinTolerance(z, last, tsquare))
            break;
        last = z;
    }
    if(i == maxIters)
        return -1;

    for(int j = 0; j < numAttractors; ++j) {
        if(withinTolerance(z, attractors[j], tsquare))
            return j;
    }
    return -1;
}