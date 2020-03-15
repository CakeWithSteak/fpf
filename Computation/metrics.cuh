#include "kernel_types.h"
#include "kernel_macros.cuh"

#include "distance_metrics/fixedpoint.cuh"

DEFER_TO_NVRTC_PREPROCESSOR #ifdef FIXEDPOINT_ITERATIONS
DEFER_TO_NVRTC_PREPROCESSOR #define DIST_F fixedPointDist
DEFER_TO_NVRTC_PREPROCESSOR #else
        DEFER_TO_NVRTC_PREPROCESSOR #error "Invalid distance metric."
DEFER_TO_NVRTC_PREPROCESSOR #endif