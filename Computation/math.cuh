#pragma once

#ifndef BUILD_FOR_NVRTC //Hack for code insights
#include <vector_functions.h>
#include <cmath>
#endif

#include "kernel_macros.cuh"
#include "kernel_stdint.cuh"
using complex = float2;

DEFER_TO_NVRTC_PREPROCESSOR #define make_complex make_float2

__device__ __inline__ complex cconj(complex z) {
    return make_complex(
            z.x,
            -z.y
    );
}

__device__ __inline__ complex cadd(complex a, complex b) {
    return make_complex(
            a.x + b.x,
            a.y + b.y
            );
}

__device__ __inline__ complex csub(complex a, complex b) {
    return make_complex(
            a.x - b.x,
            a.y - b.y
    );
}

__device__ __inline__ complex cmul(complex a, complex b) {
    return make_complex(
            (a.x * b.x) - (a.y * b.y),
            (a.x * b.y) + (b.x * a.y)
    );
}

__device__ __inline__ complex ccos(complex z) {
    return make_complex(
            cosf(z.x) * coshf(z.y),
            -1 * sinf(z.x) * sinhf(z.y)
    );
}

__device__ __inline__ complex csin(complex z) {
    return make_complex(
            sinf(z.x) * coshf(z.y),
            cosf(z.x) * sinhf(z.y)
    );
}

__device__ __inline__ complex ctan(complex z) {
    float x = cosf(2*z.x) + coshf(2*z.y);
    return make_complex(
            sinf(2*z.x) / x,
            sinhf(2*z.y) / x
    );
}

__device__ __inline__ complex cneg(complex z) {
    return make_complex(
            -z.x,
            -z.y
    );
}

__device__ __inline__ complex cnot(complex z) {
    uint64_t re = ~*reinterpret_cast<uint32_t*>(&z.x);
    uint64_t im = ~*reinterpret_cast<uint32_t*>(&z.y);
    return make_complex(
            *reinterpret_cast<float*>(&re),
            *reinterpret_cast<float*>(&im)
    );
}

__device__ __inline__ complex cor(complex a, complex b) {
    uint64_t re1 = ~*reinterpret_cast<uint32_t*>(&a.x);
    uint64_t im1 = ~*reinterpret_cast<uint32_t*>(&a.y);
    uint64_t re2 = ~*reinterpret_cast<uint32_t*>(&b.x);
    uint64_t im2 = ~*reinterpret_cast<uint32_t*>(&b.y);
    uint64_t re = re1 | re2;
    uint64_t im = im1 | im2;
    return make_complex(
            *reinterpret_cast<float*>(&re),
            *reinterpret_cast<float*>(&im)
    );
}

__device__ __inline__ complex cand(complex a, complex b) {
    uint64_t re1 = ~*reinterpret_cast<uint32_t*>(&a.x);
    uint64_t im1 = ~*reinterpret_cast<uint32_t*>(&a.y);
    uint64_t re2 = ~*reinterpret_cast<uint32_t*>(&b.x);
    uint64_t im2 = ~*reinterpret_cast<uint32_t*>(&b.y);
    uint64_t re = re1 & re2;
    uint64_t im = im1 & im2;
    return make_complex(
            *reinterpret_cast<float*>(&re),
            *reinterpret_cast<float*>(&im)
    );
}

__device__ __inline__ complex cxor(complex a, complex b) {
    uint64_t re1 = ~*reinterpret_cast<uint32_t*>(&a.x);
    uint64_t im1 = ~*reinterpret_cast<uint32_t*>(&a.y);
    uint64_t re2 = ~*reinterpret_cast<uint32_t*>(&b.x);
    uint64_t im2 = ~*reinterpret_cast<uint32_t*>(&b.y);
    uint64_t re = re1 ^ re2;
    uint64_t im = im1 ^ im2;
    return make_complex(
            *reinterpret_cast<float*>(&re),
            *reinterpret_cast<float*>(&im)
    );
}