#pragma once

// __device__ __inline__ dist_t DIST_F(complex z, int maxIters, complex p, real arg, complex c)

#ifdef BUILD_FOR_NVRTC
#include "kernel_macros.cuh"
#include "distance_metrics/fixedpoint.cuh"
#include "distance_metrics/julia.cuh"
#include "distance_metrics/fixedpoint_euclid.cuh"
#include "distance_metrics/vectorfield_mag.cuh"
#include "distance_metrics/vectorfield_angle.cuh"
#include "distance_metrics/periodic.cuh"
#include "distance_metrics/julia-complete.cuh"

RUNTIME #ifdef FIXEDPOINT_ITERATIONS
RUNTIME #define DIST_F fixedPointDist
RUNTIME #elif defined(JULIA)
RUNTIME #define DIST_F juliaDist
RUNTIME #elif defined(FIXEDPOINT_EUCLIDEAN)
RUNTIME #define DIST_F fixedPointDistEuclid
RUNTIME #elif defined(VECTORFIELD_MAGNITUDE)
RUNTIME #define DIST_F vfMagnitude
RUNTIME #elif defined(VECTORFIELD_ANGLE)
RUNTIME #define DIST_F vfAngle
RUNTIME #elif defined(ATTRACTOR)
//ATTRACTOR is a special case handled in kernel.cu
RUNTIME #elif defined(PERIODIC)
RUNTIME #define DIST_F periodic
RUNTIME #elif defined(JULIA_COMPLETE)
RUNTIME #define DIST_F juliaCompleteDist
RUNTIME #else
RUNTIME #error "Invalid distance metric."
RUNTIME #endif
#endif