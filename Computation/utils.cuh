#pragma once
#include "kernel_types.cuh"

__device__ __inline__ real distSquare(complex a, complex b) {
    real xdist = a.x - b.x;
    real ydist = a.y - b.y;
    return (xdist * xdist + ydist * ydist);
}

__device__ __inline__ bool withinTolerance(complex a, complex b, real tsquare) {
    return distSquare(a, b) <= tsquare;
}

__device__ __inline__ complex getZ(real re0, real re1, real im0, real im1, int width, int height, int x, int y) {
    real reStep = (re1 - re0) / width;
    real imStep = (im1 - im0) / height;
    return make_complex(
            re0 + reStep * x,
            im0 + imStep * (width - y)
    );
}