#pragma once
#include "../kernel_types.h"
#include "utils.cuh"

__device__ __inline__ dist_t vfMagnitude(complex z, dist_t unused1, complex p, float unused2) {
    complex next = F(z, p);
    return sqrtf(distSquare(z, next));
}