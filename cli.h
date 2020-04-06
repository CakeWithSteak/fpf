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
    std::optional<std::filesystem::path> deserializationPath = {};
};

Options getOptions(int argc, char** argv);