#pragma once
#include "../Computation/metrics.h"

// Configuration data of a given mode
struct ModeInfo {
    DistanceMetric metric;
    std::string displayName;
    std::string internalName;
    std::string argDisplayName;
    float argInitValue;
    float argStep;
    float argMin;
    float argMax;
};


