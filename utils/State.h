#pragma once
#include <string>
#include <complex>
#include "../Computation/kernel_types.h"
#include "Viewport.h"
#include "../Rendering/Window.h"
#include "../Rendering/Renderer.h"
#include "Input.h"
#include "../cli.h"
#include "../modes.h"

//A struct encompassing program state useful for event handlers and serialization
struct State {
    std::string expr;
    int maxIters;
    float metricArg;
    std::complex<float> p{0};
    Viewport viewport;
    bool colorCutoffEnabled;
    float colorCutoff;
    int width, height;
    ModeInfo mode; //Only the DistanceMetric is serialized
    std::optional<std::complex<float>> pathStart = {};

    explicit State(const Options& opt) {
        expr = opt.expression;
        mode = opt.mode;
        width = opt.width;
        height = opt.height;
        metricArg = (opt.metricArg.has_value()) ? opt.metricArg.value() : mode.argInitValue;
        colorCutoffEnabled = (mode.defaultColorCutoff != -1);
        colorCutoff = colorCutoffEnabled ? mode.defaultColorCutoff : 10.0f;
        maxIters = mode.initMaxIters;
    }
    State() = default;
};

struct RuntimeState {
    Window& window;
    Renderer& renderer;
    bool forceRerender = false;
    std::filesystem::path refsPath;
};