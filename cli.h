#pragma once
#include "utils/ModeInfo.h"
#include <string>

struct Options {
    ModeInfo mode;
    std::string expression;
    int width;
    int height;
};

Options getOptions(int argc, char** argv);