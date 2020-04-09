#pragma once
#include <filesystem>
#include <boost/serialization/access.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/string.hpp>
#include "State.h"


template <class Archive>
void serialize(Archive& ar, State& s, const unsigned int version) {
    ar & s.expr;
    ar & s.maxIters;
    ar & s.metricArg;
    ar & s.p;
    ar & s.viewport;
    ar & s.colorCutoffEnabled;
    ar & s.colorCutoff;
    ar & s.width;
    ar & s.height;
    ar & s.mode;
    ar & s.pathStart;
}

void save(State& state);
State deserialize(const std::filesystem::path& path);