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

//todo capturing?
__global__ void transformLine(float re1, float re2, float im1, float im2, float pre, float pim, int numPoints, int iteration, complex* output) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    complex z = getZOnLine(re1, re2, im1, im2, numPoints, tid);
    const complex c = z;
    const complex p = make_complex(pre, pim);

    for(int i = 0; i < iteration; ++i) {
        z = F(z, p, c);
    }
    output[tid] = z;
}
