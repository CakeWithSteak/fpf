#pragma once
#include "kernel_types.cuh"
#include "kernel_macros.cuh"
#include "shared_types.h"

__device__ __inline__ complex getZOnLine(double re1, double re2, double im1, double im2, int numPoints, int tid) {
    double f = static_cast<double>(tid) / (numPoints - 1);
    return make_complex(
            re1 + ((re2 - re1) * f),
            im1 + ((im2 - im1) * f)
    );
}

__device__ __inline__ complex getZOnCircle(double2 center, double r, int numPoints, int tid) {
    double theta = 2 * CUDART_PI * static_cast<double>(tid) / (numPoints - 1);
    return make_complex (
        center.x + r * cos(theta),
        center.y + r * sin(theta)
    );
}

__device__ __inline__ complex getZ(const ShapeProps& props, int numPoints, int tid) {
    switch(props.shape) {
        case LINE:
            return getZOnLine(props.line.p1.x, props.line.p2.x, props.line.p1.y, props.line.p2.y, numPoints, tid);
        case CIRCLE:
            return getZOnCircle(props.circle.center, props.circle.r, numPoints, tid);
        default:
            return make_complex(0,0);
    }
}

__global__ void transformShape(ShapeProps props, real pre, real pim, int numPoints, int iteration, bool incremental, double2* output) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid >= numPoints)
        return;

    complex z, c;
    if(incremental) {
        z = double2_to_complex(output[tid]);
    } else {
RUNTIME #ifdef CAPTURING
        c = getZ(props, numPoints, tid);
        z = make_complex(0, 0);
RUNTIME #else
        z = getZ(props, numPoints, tid);
        c = z;
RUNTIME #endif
    }

    const complex p = make_complex(pre, pim);

    for(int i = 0; i < iteration; ++i) {
        z = F(z, p, c);
    }
    output[tid] = complex_to_double2(z);
}
