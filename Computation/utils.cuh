#pragma once
#include "kernel_types.h"

__device__ __inline__ float distSquare(complex a, complex b) {
    float xdist = a.x - b.x;
    float ydist = a.y - b.y;
    return (xdist * xdist + ydist * ydist);
}

__device__ __inline__ bool withinTolerance(complex a, complex b, float tsquare) {
    return distSquare(a, b) <= tsquare;
}

__device__ __inline__ float2 getZ(float re0, float re1, float im0, float im1, int width, int height, int x, int y) {
    float reStep = (re1 - re0) / width;
    float imStep = (im1 - im0) / height;
    return make_complex(
            re0 + reStep * x,
            im0 + imStep * (width - y)
    );
}