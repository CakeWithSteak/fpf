#pragma once
#include "kernel_types.h"
#include "kernel_macros.cuh"

__device__ __inline__ complex getZOnLine(float re1, float re2, float im1, float im2, int numPoints, int tid) {
    float f = static_cast<float>(tid) / (numPoints - 1);
    return make_complex(
            re1 + ((re2 - re1) * f),
            im1 + ((im2 - im1) * f)
    );
}

__global__ void transformLine(float re1, float re2, float im1, float im2, float pre, float pim, int numPoints, int iteration, bool incremental, complex* output) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid >= numPoints)
        return;

    complex z, c;
    if(incremental) {
        z = output[tid];
    } else {
RUNTIME #ifdef CAPTURING
        c = getZOnLine(re1, re2, im1, im2, numPoints, tid);
        z = make_complex(0, 0);
RUNTIME #else
        z = getZOnLine(re1, re2, im1, im2, numPoints, tid);
        c = z;
RUNTIME #endif
    }

    const complex p = make_complex(pre, pim);

    for(int i = 0; i < iteration; ++i) {
        z = F(z, p, c);
    }
    output[tid] = z;
}
