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
    int maxIters = 128;
    float metricArg;
    std::complex<float> p{0};
    Viewport viewport;
    bool colorCutoffEnabled;
    float colorCutoff;
    int width, height;
    ModeInfo mode; //Only the DistanceMetric is serialized

    State(const Options& opt) {
        expr = opt.expression;
        mode = opt.mode;
        width = opt.width;
        height = opt.height;
        metricArg = mode.argInitValue;
        colorCutoffEnabled = (mode.defaultColorCutoff != -1);
        colorCutoff = colorCutoffEnabled ? mode.defaultColorCutoff : 10.0f;
    }
    State() = default;
};

struct RuntimeState {
    Window& window;
    Renderer& renderer;
};