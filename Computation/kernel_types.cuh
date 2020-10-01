#pragma once
#include "kernel_macros.cuh"
#include <float.h>

using dist_t = float; // Single precision floats have sufficed here so far, no point in complicating this further
using dist2 = float2;
#define make_dist2 make_float2

RUNTIME #ifdef PREC_FLOAT

using real = float;
using complex = float2;
RUNTIME #define EPSILON FLT_EPSILON
RUNTIME #define make_complex make_float2
RUNTIME #define complex_to_float2(x) x
RUNTIME #define float2_to_complex(x) x
__device__ __inline__ double2 complex_to_double2(complex z) {
    return make_double2(z.x, z.y);
}
__device__ __inline__ complex double2_to_complex(double2 z){
    return make_complex(z.x, z.y);
}

RUNTIME #elif defined(PREC_DOUBLE)

using complex = double2;
RUNTIME #define make_complex make_double2
using real = double;
RUNTIME #define EPSILON DBL_EPSILON
__device__ __inline__ float2 complex_to_float2(complex z) {
    return make_float2(z.x, z.y);
}
__device__ __inline__ complex float2_to_complex(float2 z) {
    return make_complex(z.x, z.y);
}
RUNTIME #define complex_to_double2(x) x
RUNTIME #define double2_to_complex(x) x

RUNTIME #endif

__device__ __inline__ complex F(complex z, complex p, complex c); //Forward declaration for generated code