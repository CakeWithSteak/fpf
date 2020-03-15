#pragma once
#ifndef BUILD_FOR_NVRTC
#include <builtin_types.h>
#endif

using dist_t = int;
using dist2 = int2;
#define make_dist2 make_int2
using complex = float2;

#ifndef BUILD_FOR_NVRTC
#include <map>

enum DistanceMetric {
    FIXEDPOINT_ITERATIONS
};

const std::map<DistanceMetric, std::string> metricMacroMap {
        {FIXEDPOINT_ITERATIONS, "FIXEDPOINT_ITERATIONS"}
};

#else
__device__ __inline__ complex F(complex z, complex p); //Forward declaration for generated code
#endif