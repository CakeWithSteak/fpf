#pragma once
#include "kernel_types.h"

// __device__ __inline__ DIST_F(complex z, dist_t maxIters, complex p, float arg, complex c)

#ifdef BUILD_FOR_NVRTC
#include "kernel_macros.cuh"
#include "distance_metrics/fixedpoint.cuh"
#include "distance_metrics/julia.cuh"
#include "distance_metrics/fixedpoint_euclid.cuh"
#include "distance_metrics/vectorfield_mag.cuh"
#include "distance_metrics/vectorfield_angle.cuh"

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
RUNTIME #else
RUNTIME #error "Invalid distance metric."
RUNTIME #endif
#else
#include <map>

enum DistanceMetric {
    FIXEDPOINT_ITERATIONS,
    FIXEDPOINT_EUCLIDEAN,
    JULIA,
    VECTORFIELD_MAGNITUDE,
    VECTORFIELD_ANGLE,

    CAPTURING_JULIA,
    CAPTURING_FIXEDPOINT
};

inline float prepMetricArg(DistanceMetric metric, float arg) {
    if(metric == FIXEDPOINT_ITERATIONS || metric == JULIA || metric == FIXEDPOINT_EUCLIDEAN)
        return arg * arg;
    return arg;
}

#endif