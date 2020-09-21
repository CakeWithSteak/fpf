#pragma once
#include <utility>
#include <complex>
#include <optional>

template <typename T>
using Interpolate = std::pair<T, T>;

struct AnimationParams {
    double duration;
    int fps;

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

    [[nodiscard]] int totalFrames() const {
        return std::ceil(fps * duration);
    }

    std::string getReferenceString(std::string_view func) {
        std::stringstream ss;
        ss << func << "\t"
           << duration << "\t"
           << fps << "\t"
           << maxIters.first << " -> " << maxIters.second << "\t"
           << metricArg.first << " -> " << metricArg.second << "\t"
           << p.first << " -> " << p.second << "\t"
           << viewportCenter.first << " -> "<< viewportCenter.second << "\t"
           << viewportBreadth.first << " -> "<< viewportBreadth.second << "\t"
           << (colorCutoff ? colorCutoff->first : -1) << " -> " << (colorCutoff ? colorCutoff->second : -1) << "\t";

        if(pathStart.has_value())
            ss << pathStart->first << " -> " << pathStart->second << "\t";
        else
            ss << "NONE\t";
        if(lineTransStart.has_value()) {
            ss << lineTransStart->first << " -> " << lineTransStart->second << "\t"
               << lineTransEnd->first << " -> " << lineTransEnd->second << "\t"
               << lineTransIteration.first << " -> " << lineTransIteration.second;
        }
        else {
            ss << "NONE\tNONE\tNONE";
        }
        ss << "\n";
        return ss.str();
    }
};
