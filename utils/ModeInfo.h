#pragma once
#include "../Computation/metrics.h"

//todo different coloroing for different modes
// Configuration data of a given mode
struct ModeInfo {
    DistanceMetric metric;
    std::string displayName;
    std::string cliName;
    std::string internalName;
    std::string argDisplayName;
    float argInitValue;
    float argStep;
    float argMin;
    float argMax;
    bool disableArg = false;
    bool disableIterations = false;
};


