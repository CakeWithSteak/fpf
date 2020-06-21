#pragma once
#include "utils/Input.h"
#include "utils/State.h"
#include "utils/serialization.h"
#include "utils/imageExport.h"

InputHandler initControls(State& s, RuntimeState& rs) {
    constexpr double MOVE_STEP = 0.8f;
    constexpr double ZOOM_STEP = 0.4f;
    constexpr int ITER_STEP = 2;
    constexpr double PARAM_STEP = 0.05f;
    constexpr double COLOR_CUTOFF_STEP = 1.0f;
    constexpr double FAST_MODE_MULTIPLIER = 6.f;

    InputHandler in(rs.window, GLFW_KEY_LEFT_SHIFT, FAST_MODE_MULTIPLIER);
    in.addViewport(s.viewport, GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_KP_ADD, GLFW_KEY_KP_SUBTRACT, GLFW_KEY_HOME, MOVE_STEP, ZOOM_STEP);
    auto& mode = s.mode;

    if(!mode.disableIterations)
        in.addScalar(s.maxIters, GLFW_KEY_KP_MULTIPLY, GLFW_KEY_KP_DIVIDE, ITER_STEP, "Max iterations", 1, 1024);

    if(!mode.disableArg)
        in.addScalar(s.metricArg, GLFW_KEY_0, GLFW_KEY_9, mode.argStep, mode.argDisplayName, mode.argMin, mode.argMax);

    in.addScalar(s.p, GLFW_KEY_D, GLFW_KEY_A, {PARAM_STEP}, "Parameter");
    in.addScalar(s.p, GLFW_KEY_W, GLFW_KEY_S, {0, PARAM_STEP}, "Parameter");
    in.addScalar(s.colorCutoff, GLFW_KEY_C, GLFW_KEY_X, COLOR_CUTOFF_STEP, "Color cutoff", 0.0);
    in.addToggle(s.colorCutoffEnabled, GLFW_KEY_Z, "Color cutoff");
    in.addTrigger([&rs](){rs.window.setShouldClose(true);}, GLFW_KEY_ESCAPE);
    in.addTrigger([&s, &rs](){rs.window.minimize(); save(s); rs.window.restore();}, GLFW_KEY_TAB);

    if(!mode.disableOverlays) {
        rs.mouseBinding = &in.addTrigger([&s, &rs](double x, double y) {
            auto z = s.viewport.resolveScreenCoords(x, y, rs.window.getWidth(), rs.window.getHeight());
            if(!s.lineTransEnabled) { //Path mode
                s.pathStart = z;
                std::cout << "Rendering path overlay from " << z << ".\n";
                auto length = rs.renderer.generatePath(z, s.metricArg, s.p);
                std::cout << "Path length: " << length << std::endl;
            } else { // Line transform mode
                if(!s.lineTransStart.has_value()) { //First click to select start point
                    s.lineTransStart = z;
                    std::cout << "Set line transform start point to" << *s.lineTransStart << "." << std::endl;
                } else if(!s.lineTransEnd.has_value()) { //Second click to select endpoint
                    s.lineTransEnd = z;
                    rs.renderer.generateLineTransform(*s.lineTransStart, *s.lineTransEnd, s.lineTransIteration, s.p);
                    std::cout << "Rendering line transform overlay from " << *s.lineTransStart << " to " << *s.lineTransEnd
                        << "." << std::endl;
                }
            }
        }, GLFW_MOUSE_BUTTON_1);

        in.addTrigger([&s, &rs]() {
            rs.renderer.hideOverlay();
            s.pathStart = {}; s.lineTransStart = {}; s.lineTransEnd = {};
            s.lineTransEnabled = false;
            rs.mouseBinding->setCooldown(0ms); // If we were in line trans mode we need to unset the cooldown
        }, GLFW_KEY_H);

        in.addTrigger([&s, &rs]() {
            if(s.lineTransEnabled)
                 return;
            rs.renderer.hideOverlay();
            s.lineTransEnabled = true;
            std::cout << "Line transform mode enabled. Click two points to start." << std::endl;
            s.lineTransIteration = 0;
            rs.mouseBinding->setCooldown(200ms);
        }, GLFW_KEY_T);

        in.addTrigger([&s, &rs]() {
            if(s.lineTransEnabled && s.lineTransEnd.has_value()) {
                ++s.lineTransIteration;
                rs.renderer.setLineTransformIteration(s.lineTransIteration, s.p, s.forceDisableIncrementalLineTracing);
                std::cout << "Line transform iteration: " << s.lineTransIteration << "." << std::endl;
            }
        }, GLFW_KEY_RIGHT_SHIFT).setCooldown(150ms);

        in.addTrigger([&s, &rs]() {
            if(s.lineTransEnabled && s.lineTransEnd.has_value() && s.lineTransIteration != 0) {
                --s.lineTransIteration;
                rs.renderer.setLineTransformIteration(s.lineTransIteration, s.p, s.forceDisableIncrementalLineTracing);
                std::cout << "Line transform iteration: " << s.lineTransIteration << "." << std::endl;
            }
        }, GLFW_KEY_RIGHT_CONTROL).setCooldown(150ms);

        in.addTrigger([&rs]() { rs.renderer.togglePointConnections();}, GLFW_KEY_O).setCooldown(200ms);
    }

    in.addTrigger([&s, &rs](){
        rs.window.minimize();
        std::cout << "Save image as> ";
        std::filesystem::path filename;
        std::cin >> filename;
        filename.replace_extension("png");
        auto pixels = rs.renderer.exportImageData();
        exportImage(filename, rs.window.getWidth(), rs.window.getHeight(), pixels);
        writeImageInfoToReferences(rs.refsPath, filename, s);
        rs.window.restore();
    }, GLFW_KEY_INSERT);

    return in;
}