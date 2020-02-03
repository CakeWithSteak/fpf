#pragma once
#include <vector_functions.h>
#include <cuComplex.h>
#include <cstdint>

using complex = cuFloatComplex;

//Some macros for functions defined in cuComplex.h
#define make_complex make_float2
#define cconj cuConjf
#define cadd cuCaddf
#define csub cuCsubf
#define cmul cuCmulf
#define cdiv cuCdivf
#define cabs cuCabsf

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