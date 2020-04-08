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