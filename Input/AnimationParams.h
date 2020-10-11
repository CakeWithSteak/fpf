#pragma once
#include <utility>
#include <complex>
#include <optional>
#include <cassert>
#include "../Computation/shared_types.h"

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
    std::optional<Interpolate<ShapeProps>> shapeProps;
    Interpolate<int> shapeTransIteration;

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
        if(shapeProps.has_value()) {
            assert(shapeProps->first.shape == shapeProps->second.shape);
            switch(shapeProps->first.shape) {
                case LINE:
                    ss << "Line\t"
                       << shapeProps->first.line.p1.std() << " -> " << shapeProps->second.line.p1.std() << "\t"
                       << shapeProps->first.line.p2.std() << " -> " << shapeProps->second.line.p2.std() << "\t";
                    break;
                case CIRCLE:
                    ss << "Circle\t"
                       << shapeProps->first.circle.center.std() << " -> " << shapeProps->second.circle.center.std() << "\t"
                       << shapeProps->first.circle.r << " -> " << shapeProps->second.circle.r << "\t";
                    break;
                default:
                    throw std::runtime_error("Invalid tag on ShapeProps");
            }
            ss << shapeTransIteration.first << " -> " << shapeTransIteration.second;
        }
        else {
            ss << "NONE";
        }
        ss << "\n";
        return ss.str();
    }
};
