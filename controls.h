#pragma once
#include "utils/Input.h"
#include "utils/ModeInfo.h"

InputHandler initControls(Window& window,
                          Viewport& viewport,
                          const ModeInfo& mode,
                          int& maxIters, float& metricArg, std::complex<float>& p, bool& colorCutoffEnabled, float& colorCutoff
                          ) {
    constexpr float MOVE_STEP = 0.8f;
    constexpr float ZOOM_STEP = 0.4f;
    constexpr int ITER_STEP = 2;
    constexpr float PARAM_STEP = 0.05f;
    constexpr float COLOR_CUTOFF_STEP = 1.0f;

    InputHandler in(window);
    in.addViewport(viewport, GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_KP_ADD, GLFW_KEY_KP_SUBTRACT, GLFW_KEY_HOME, MOVE_STEP, ZOOM_STEP);

    if(!mode.disableIterations)
        in.addScalar(maxIters, GLFW_KEY_KP_MULTIPLY, GLFW_KEY_KP_DIVIDE, ITER_STEP, "Max itertations", 1, 1024);

    if(!mode.disableArg)
        in.addScalar(metricArg, GLFW_KEY_0, GLFW_KEY_9, mode.argStep, mode.argDisplayName, mode.argMin, mode.argMax);

    in.addScalar(p, GLFW_KEY_D, GLFW_KEY_A, {PARAM_STEP}, "Parameter");
    in.addScalar(p, GLFW_KEY_W, GLFW_KEY_S, {0, PARAM_STEP}, "Parameter");
    in.addScalar(colorCutoff, GLFW_KEY_C, GLFW_KEY_X, COLOR_CUTOFF_STEP, "Color cutoff", 0.0f);
    in.addToggle(colorCutoffEnabled, GLFW_KEY_Z, "Color cutoff");
    return in;
}