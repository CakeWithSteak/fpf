#pragma once
#include "modes.h"
#include <string>
#include <filesystem>
#include <optional>

struct Options {
    ModeInfo mode;
    std::string expression;
    int width;
    int height;
    std::filesystem::path refsPath;
    std::optional<float> metricArg;
    std::optional<std::filesystem::path> deserializationPath = {};
    bool forceDisableIncrementalLineTracing;
    bool doublePrec;
};

Options getOptions(int argc, char** argv);