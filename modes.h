#pragma once
#include "utils/ModeInfo.h"

const std::map<DistanceMetric, ModeInfo> modes{
        {FIXEDPOINT_ITERATIONS, {
            .metric = FIXEDPOINT_ITERATIONS,
            .displayName = "Fixed point distance (iterations)",
            .internalName = "FIXEDPOINT_ITERATIONS",
            .argDisplayName = "Tolerance",
            .argInitValue = 0.01f,
            .argStep = 0.0001f,
            .argMin = 0.0f,
            .argMax = 2.0f
        }},
        {JULIA, {
            .metric = JULIA,
            .displayName = "Julia set",
            .internalName = "JULIA",
            .argDisplayName = "Escape radius",
            .argInitValue = 10.0f,
            .argStep = 0.05f,
            .argMin = 0.0f,
            .argMax = 200.0f
        }}
};