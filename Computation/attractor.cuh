#include "kernel_types.cuh"
#include "utils.cuh"
#include "constants.h"

RUNTIME #ifdef PREC_FLOAT
RUNTIME #define NAN (0.0f / 0.0f)
RUNTIME #elif defined(PREC_DOUBLE)
RUNTIME #define NAN (0.0 / 0.0)
RUNTIME #endif

__global__ void findAttractors(real re0, real re1, real im0, real im1, int maxIters, real pre, real pim, real tsquare, int width, int height, float2* output) {
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
            output[tid] = complex_to_float2(z);
            return;
        }
        last = z;
    }
    output[tid] = make_float2(NAN, NAN);
}

__device__ __inline__ dist_t whichAttractor(complex z, int maxIters, complex p, real tsquare, complex c, const float2* attractors, size_t numAttractors) {
    complex last = z;
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
        if(withinTolerance(z, float2_to_complex(attractors[j]), KERNEL_ATTRACTOR_MAX_TOL))
            return j;
    }
    return -1;
}