#pragma once
#include <utility>
#include <complex>
#include <optional>

template <typename T>
using Interpolate = std::pair<T, T>;

struct AnimationParams {
    double duration;
    int fps;
    //todo different interpolation modes

    Interpolate<int> maxIters;
    Interpolate<double> metricArg;
    Interpolate<std::complex<double>> p;
    Interpolate<std::complex<double>> viewportCenter;
    Interpolate<double> viewportBreadth;
    std::optional<Interpolate<double>> colorCutoff;
    std::optional<Interpolate<std::complex<double>>> pathStart;
    std::optional<Interpolate<std::complex<double>>> lineTransStart;
    std::optional<Interpolate<std::complex<double>>> lineTransEnd;
    Interpolate<int> lineTransIteration;
};