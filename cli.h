#pragma once
#include "modes.h"
#include "Input/AnimationParams.h"
#include <string>
#include <filesystem>
#include <optional>
#include <complex>

struct Options {
    ModeInfo mode;
    std::string expression;
    int width;
    int height;
    int maxIters;
    std::optional<std::filesystem::path> refsPath;
    double metricArg;
    std::optional<std::filesystem::path> deserializationPath = {};
    bool forceDisableIncrementalShapeTrans;
    bool doublePrec;
    std::complex<double> p = 0;
    std::complex<double> viewportCenter = 0;
    double viewportBreadth = 2;
    std::optional<double> colorCutoff;
    std::optional<AnimationParams> animParams;
    std::optional<std::filesystem::path> animPath;
    bool animBackground;
    std::filesystem::path cudaIncludePath;
    bool enableVsync;
};

Options getOptions(int argc, char** argv);