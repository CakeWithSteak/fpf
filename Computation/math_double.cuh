#pragma once

#include "kernel_macros.cuh"
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

__device__ __inline__ complex cln(complex z) {
    return make_complex(log(sqrt(z.x*z.x + z.y*z.y)), atan2(z.y, z.x));
}

__device__ __inline__ complex cpow(complex a, complex b) {
    return cexp(cmul(b, cln(a)));
}