#pragma once
#include "utils/Input.h"
#include "utils/State.h"
#include "utils/serialization.h"
#include "utils/imageExport.h"

InputHandler initControls(State& s, RuntimeState& rs) {
    constexpr float MOVE_STEP = 0.8f;
    constexpr float ZOOM_STEP = 0.4f;
    constexpr int ITER_STEP = 2;
    constexpr float PARAM_STEP = 0.05f;
    constexpr float COLOR_CUTOFF_STEP = 1.0f;


    InputHandler in(rs.window);
    in.addViewport(s.viewport, GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_KP_ADD, GLFW_KEY_KP_SUBTRACT, GLFW_KEY_HOME, MOVE_STEP, ZOOM_STEP);
    auto& mode = s.mode;

    if(!mode.disableIterations)
        in.addScalar(s.maxIters, GLFW_KEY_KP_MULTIPLY, GLFW_KEY_KP_DIVIDE, ITER_STEP, "Max itertations", 1, 1024);

    if(!mode.disableArg)
        in.addScalar(s.metricArg, GLFW_KEY_0, GLFW_KEY_9, mode.argStep, mode.argDisplayName, mode.argMin, mode.argMax);

    in.addScalar(s.p, GLFW_KEY_D, GLFW_KEY_A, {PARAM_STEP}, "Parameter");
    in.addScalar(s.p, GLFW_KEY_W, GLFW_KEY_S, {0, PARAM_STEP}, "Parameter");
    in.addScalar(s.colorCutoff, GLFW_KEY_C, GLFW_KEY_X, COLOR_CUTOFF_STEP, "Color cutoff", 0.0f);
    in.addToggle(s.colorCutoffEnabled, GLFW_KEY_Z, "Color cutoff");
    in.addTrigger([&rs](){rs.window.setShouldClose(true);}, GLFW_KEY_ESCAPE);
    in.addTrigger([&s, &rs](){rs.window.minimize(); save(s); rs.window.restore();}, GLFW_KEY_TAB);

    in.addTrigger([&s, &rs](double x, double y){
        auto z = s.viewport.resolveScreenCoords(x, y, rs.window.getWidth(), rs.window.getHeight());
        s.pathStart = z;
        std::cout << "Rendering path overlay from " << z << ".\n";
        auto length = rs.renderer.generatePath(z, s.metricArg, s.p);
        std::cout << "Path length: " << length << std::endl;
    }, GLFW_MOUSE_BUTTON_1);

    in.addTrigger([&s, &rs](){rs.renderer.hidePath(); s.pathStart = {};}, GLFW_KEY_H);

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