#pragma once

#include "kernel_macros.cuh"
#include "kernel_stdint.cuh"
#include "kernel_types.cuh"

RUNTIME #include <cuComplex.h>

//Aliases for functions defined in cuComplex
RUNTIME #define cadd cuCadd
RUNTIME #define csub cuCsub
RUNTIME #define cmul cuCmul
RUNTIME #define cdiv cuCdiv
RUNTIME #define cconj cuConj

__device__ __inline__ complex ccos(complex z) {
    return make_complex(
            cos(z.x) * cosh(z.y),
            -1 * sin(z.x) * sinh(z.y)
    );
}

__device__ __inline__ complex csin(complex z) {
    return make_complex(
            sin(z.x) * cosh(z.y),
            cos(z.x) * sinh(z.y)
    );
}

__device__ __inline__ complex ctan(complex z) {
    double x = cos(2*z.x) + cosh(2*z.y);
    return make_complex(
            sin(2*z.x) / x,
            sinh(2*z.y) / x
    );
}

__device__ __inline__ complex cneg(complex z) {
    return make_complex(
            -z.x,
            -z.y
    );
}

/*
__device__ __inline__ complex cnot(complex z) {
    uint64_t re = ~*reinterpret_cast<uint64_t*>(&z.x);
    uint64_t im = ~*reinterpret_cast<uint64_t*>(&z.y);
    return make_complex(
            *reinterpret_cast<double*>(&re),
            *reinterpret_cast<double*>(&im)
    );
}

__device__ __inline__ complex cor(complex a, complex b) {
    uint64_t re1 = ~*reinterpret_cast<uint64_t*>(&a.x);
    uint64_t im1 = ~*reinterpret_cast<uint64_t*>(&a.y);
    uint64_t re2 = ~*reinterpret_cast<uint64_t*>(&b.x);
    uint64_t im2 = ~*reinterpret_cast<uint64_t*>(&b.y);
    uint64_t re = re1 | re2;
    uint64_t im = im1 | im2;
    return make_complex(
            *reinterpret_cast<double*>(&re),
            *reinterpret_cast<double*>(&im)
    );
}

__device__ __inline__ complex cand(complex a, complex b) {
    uint64_t re1 = ~*reinterpret_cast<uint64_t*>(&a.x);
    uint64_t im1 = ~*reinterpret_cast<uint64_t*>(&a.y);
    uint64_t re2 = ~*reinterpret_cast<uint64_t*>(&b.x);
    uint64_t im2 = ~*reinterpret_cast<uint64_t*>(&b.y);
    uint64_t re = re1 & re2;
    uint64_t im = im1 & im2;
    return make_complex(
            *reinterpret_cast<double*>(&re),
            *reinterpret_cast<double*>(&im)
    );
}

__device__ __inline__ complex cxor(complex a, complex b) {
    uint64_t re1 = ~*reinterpret_cast<uint64_t*>(&a.x);
    uint64_t im1 = ~*reinterpret_cast<uint64_t*>(&a.y);
    uint64_t re2 = ~*reinterpret_cast<uint64_t*>(&b.x);
    uint64_t im2 = ~*reinterpret_cast<uint64_t*>(&b.y);
    uint64_t re = re1 ^ re2;
    uint64_t im = im1 ^ im2;
    return make_complex(
            *reinterpret_cast<double*>(&re),
            *reinterpret_cast<double*>(&im)
    );
}*/ //todo remove these

__device__ __inline__ complex cabs(complex z) {
    return make_complex(cuCabs(z), 0);
}

__device__ __inline__ complex creal(complex z) {
    return make_complex(cuCreal(z), 0);
}

__device__ __inline__ complex cimag(complex z) {
    return make_complex(cuCimag(z), 0);
}

__device__ __inline__ complex carg(complex z) {
    return make_complex(atan2(z.y, z.x), 0);
}

__device__ __inline__ complex cexp(complex z) {
    double temp = exp(z.x);
    return make_complex(temp * cos(z.y), temp * sin(z.y));
}