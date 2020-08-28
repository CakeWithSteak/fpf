#pragma once

#include "kernel_macros.cuh"
#include "kernel_types.cuh"

RUNTIME #include <cuComplex.h>

//Aliases for functions defined in cuComplex
RUNTIME #define cadd cuCaddf
RUNTIME #define csub cuCsubf
RUNTIME #define cmul cuCmulf
RUNTIME #define cdiv cuCdivf
RUNTIME #define cconj cuConjf

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

__device__ __inline__ complex cabs(complex z) {
    return make_complex(cuCabsf(z), 0);
}

__device__ __inline__ complex creal(complex z) {
    return make_complex(cuCrealf(z), 0);
}

__device__ __inline__ complex cimag(complex z) {
    return make_complex(cuCimagf(z), 0);
}

__device__ __inline__ complex carg(complex z) {
    return make_complex(atan2(z.y, z.x), 0);
}

__device__ __inline__ complex cexp(complex z) {
    float temp = expf(z.x);
    return make_complex(temp * cosf(z.y), temp * sinf(z.y));
}

__device__ __inline__ complex cln(complex z) {
    return make_complex(logf(sqrtf(z.x*z.x + z.y*z.y)), atan2f(z.y, z.x));
}

__device__ __inline__ complex cpow(complex a, complex b) {
    return cexp(cmul(b, cln(a)));
}