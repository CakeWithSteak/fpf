#pragma once
#include <boost/serialization/split_member.hpp>
#include "../Computation/metrics.h"
#include <string>

struct ModeInfo;

extern const std::map<DistanceMetric, ModeInfo> modes;

struct ModeInfo {
    DistanceMetric metric;
    DistanceMetric serializedName = metric;
    std::string displayName;
    std::string cliName;
    std::string metricInternalName;
    std::string argDisplayName;
    double argInitValue;
    double argStep;
    double argMin;
    double argMax;
    bool argIsTolerance = false; // Used for path tracing
    float defaultColorCutoff = -1;
    bool disableArg = false;
    bool disableIterations = false;
    float maxHue = 0.8f;
    std::optional<std::pair<float, float>> staticMinMax = {};
    bool capturing = false;
    bool disableOverlays = false;
    int initMaxIters = 128;
    bool isAttractor = false;

    template<class Archive>
    void save(Archive& ar, const unsigned int version) const
    {
        ar << serializedName;
    }

    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        DistanceMetric m;
        ar >> m;
        *this = modes.at(m);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};