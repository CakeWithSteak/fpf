#pragma once
#include "kernel_types.h"

// __device__ __inline__ DIST_F(complex z, dist_t maxIters, complex p, float arg)

#ifdef BUILD_FOR_NVRTC
#include "kernel_macros.cuh"
#include "distance_metrics/fixedpoint.cuh"
#include "distance_metrics/julia.cuh"
#include "distance_metrics/fixedpoint_euclid.cuh"
#include "distance_metrics/vectorfield_mag.cuh"

DEFER_TO_NVRTC_PREPROCESSOR #ifdef FIXEDPOINT_ITERATIONS
DEFER_TO_NVRTC_PREPROCESSOR #define DIST_F fixedPointDist
DEFER_TO_NVRTC_PREPROCESSOR #elif defined(JULIA)
DEFER_TO_NVRTC_PREPROCESSOR #define DIST_F juliaDist
DEFER_TO_NVRTC_PREPROCESSOR #elif defined(FIXEDPOINT_EUCLIDEAN)
DEFER_TO_NVRTC_PREPROCESSOR #define DIST_F fixedPointDistEuclid
DEFER_TO_NVRTC_PREPROCESSOR #elif defined(VECTORFIELD_MAGNITUDE)
DEFER_TO_NVRTC_PREPROCESSOR #define DIST_F vfMagnitude
DEFER_TO_NVRTC_PREPROCESSOR #else
DEFER_TO_NVRTC_PREPROCESSOR #error "Invalid distance metric."
DEFER_TO_NVRTC_PREPROCESSOR #endif
#else
#include <map>

enum DistanceMetric {
    FIXEDPOINT_ITERATIONS,
    FIXEDPOINT_EUCLIDEAN,
    JULIA,
    VECTORFIELD_MAGNITUDE
};

inline float prepMetricArg(DistanceMetric metric, float arg) {
    if(metric == FIXEDPOINT_ITERATIONS || metric == JULIA || metric == FIXEDPOINT_EUCLIDEAN)
        return arg* arg;
    return arg;
}

#endif