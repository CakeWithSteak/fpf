#pragma once
#include "../kernel_types.cuh"
#include "../utils.cuh"

__device__ __inline__ dist_t vfMagnitude(complex z, dist_t unused1, complex p, real unused2, complex c) {
    complex next = F(z, p, c);
    return sqrtf(distSquare(z, next));
}