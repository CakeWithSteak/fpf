#pragma once
#include "../Computation/metrics.h"

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
    float defaultColorCutoff = -1;
    bool disableArg = false;
    bool disableIterations = false;
    float maxHue = 0.8f;
};


