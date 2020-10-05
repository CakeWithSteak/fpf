#pragma once
#include "Input/Input.h"
#include "utils/State.h"
#include "utils/serialization.h"
#include "utils/imageExport.h"

std::unique_ptr<InputHandler> initControls(State& s, RuntimeState& rs) {
    constexpr double MOVE_STEP = 0.8f;
    constexpr double ZOOM_STEP = 0.4f;
    constexpr int ITER_STEP = 2;
    constexpr double PARAM_STEP = 0.05f;
    constexpr double COLOR_CUTOFF_STEP = 1.0f;
    constexpr double FAST_MODE_MULTIPLIER = 6.f;

    std::cout << std::setprecision(12);
    
    auto in = std::make_unique<InputHandler>(rs.window, GLFW_KEY_LEFT_SHIFT, FAST_MODE_MULTIPLIER);
    in->addViewport(s.viewport, GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_KP_ADD, GLFW_KEY_KP_SUBTRACT, GLFW_KEY_HOME, MOVE_STEP, ZOOM_STEP);
    auto& mode = s.mode;

    if(!mode.disableIterations)
        in->addScalar(s.maxIters, GLFW_KEY_KP_MULTIPLY, GLFW_KEY_KP_DIVIDE, ITER_STEP, "Max iterations", 1, 2048);

    if(!mode.disableArg)
        in->addScalar(s.metricArg, GLFW_KEY_0, GLFW_KEY_9, mode.argStep, mode.argDisplayName, mode.argMin, mode.argMax);

    in->addScalar(s.p, GLFW_KEY_D, GLFW_KEY_A, {PARAM_STEP}, "Parameter");
    in->addScalar(s.p, GLFW_KEY_W, GLFW_KEY_S, {0, PARAM_STEP}, "Parameter");
    in->addScalar(s.colorCutoff, GLFW_KEY_C, GLFW_KEY_X, COLOR_CUTOFF_STEP, "Color cutoff", 0.0);
    in->addToggle(s.colorCutoffEnabled, GLFW_KEY_Z, "Color cutoff");
    in->addTrigger([&rs](){rs.window.setShouldClose(true);}, GLFW_KEY_ESCAPE);
    in->addTrigger([&s, &rs](){rs.window.minimize(); save(s); rs.window.restore();}, GLFW_KEY_TAB);

    if(!mode.disableOverlays) {
        rs.mouseBinding = &in->addTrigger([&s, &rs](double x, double y) {
            auto z = s.viewport.resolveScreenCoords(x, y, rs.window.getWidth(), rs.window.getHeight());
            if(!rs.shapeTransUIStarted && !rs.shapeTransUIFinished) { //Path mode
                s.pathStart = z;
                std::cout << "Rendering path overlay from " << z << ".\n";
                auto length = rs.renderer.generatePath(z, s.metricArg, s.p);
                std::cout << "Path length: " << length << std::endl;
            } else if(rs.selectedShape.has_value() && rs.shapeTransUIStarted){ // Shape transform mode
                if(rs.selectedShape == LINE) {
                    if (!rs.lineTransStart.has_value()) { //First click to select start point
                        rs.lineTransStart = z;
                        std::cout << "Set line transform start point to" << *rs.lineTransStart << "." << std::endl;
                    } else { //Second click to select endpoint
                        s.shapeTransProps = ShapeProps{LINE, {.line = {.p1 = *rs.lineTransStart, .p2 = z}}};
                        rs.renderer.generateShapeTransform(*s.shapeTransProps, s.shapeTransIteration, s.p);
                        std::cout << "Rendering line transform from " << *rs.lineTransStart << " to " << z
                                  << "." << std::endl;
                        rs.lineTransStart = {}; // Unset no longer needed state here to avoid having to do it somewhere else later
                        rs.shapeTransUIStarted = false;
                        rs.shapeTransUIFinished = true;
                    }
                } else if(rs.selectedShape == CIRCLE) {
                    if(!rs.circleCenter.has_value()) {
                        rs.circleCenter = z;
                        std::cout << "Set circle center to " << z << std::endl;
                    } else {
                        double r = std::abs(z - *rs.circleCenter);
                        s.shapeTransProps = ShapeProps{CIRCLE, {.circle = {*rs.circleCenter, r}}};
                        rs.renderer.generateShapeTransform(*s.shapeTransProps, s.shapeTransIteration, s.p);
                        std::cout << "Rendering circle transform from " << *rs.circleCenter << ", r = " << r
                                  << "." << std::endl;
                        rs.circleCenter = {}; // Unset no longer needed state here to avoid having to do it somewhere else later
                        rs.shapeTransUIStarted = false;
                        rs.shapeTransUIFinished = true;
                    }
                }
            }
        }, GLFW_MOUSE_BUTTON_1);

        in->addTrigger([&s, &rs]() {
            rs.renderer.hideOverlay();
            s.pathStart = {};
            rs.shapeTransUIStarted = false; rs.shapeTransUIFinished = false; rs.selectedShape = {};
            rs.mouseBinding->setCooldown(0ms); // If we were in line trans mode we need to unset the cooldown
        }, GLFW_KEY_H);

        in->addTrigger([&s, &rs]() {
            if(rs.shapeTransUIStarted || rs.shapeTransUIFinished)
                 return;
            rs.renderer.hideOverlay();
            rs.shapeTransUIStarted = true;
            s.shapeTransIteration = 0;
            std::cout << "Shape transform mode enabled. Select a shape:\n"
                << "1: Line\n"
                << "2: Circle" << std::endl; //todo maybe there's a better way to print this
            rs.mouseBinding->setEnabled(false);
        }, GLFW_KEY_T);

        in->addTrigger([&s, &rs]() {
            if(rs.shapeTransUIFinished) {
                ++s.shapeTransIteration;
                rs.renderer.setShapeTransformIteration(s.shapeTransIteration, s.p, s.forceDisableIncrementalShapeTrans);
                std::cout << "Shape transform iteration: " << s.shapeTransIteration << "." << std::endl;
            }
        }, GLFW_KEY_RIGHT_SHIFT).setCooldown(150ms);

        in->addTrigger([&s, &rs]() {
            if(rs.shapeTransUIFinished && s.shapeTransIteration != 0) {
                --s.shapeTransIteration;
                rs.renderer.setShapeTransformIteration(s.shapeTransIteration, s.p, s.forceDisableIncrementalShapeTrans);
                std::cout << "Shape transform iteration: " << s.shapeTransIteration << "." << std::endl;
            }
        }, GLFW_KEY_RIGHT_CONTROL).setCooldown(150ms);

        in->addTrigger([&rs]() { rs.renderer.togglePointConnections();}, GLFW_KEY_O).setCooldown(200ms);
    }

    in->addTrigger([&s, &rs](){
        rs.window.minimize();
        std::cout << "Save image as> ";
        std::filesystem::path filename;
        std::cin >> filename;
        filename.replace_extension("png");
        auto pixels = rs.renderer.exportImageData();
        exportImage(filename, rs.window.getWidth(), rs.window.getHeight(), pixels);
        if(rs.refsPath.has_value())
            writeImageInfoToReferences(*rs.refsPath, filename, s);
        rs.window.restore();
    }, GLFW_KEY_INSERT);

    //fixme create these bindings automatically
    //Shape trans 1 (line)
    in->addTrigger([&s, &rs](){
        if(!rs.shapeTransUIStarted || rs.selectedShape.has_value())
            return;
        rs.selectedShape = LINE;
        std::cout << "Selected line. Click two points to draw a line." << std::endl;
        rs.mouseBinding->setEnabled(true);
        rs.mouseBinding->setCooldown(200ms);
    }, GLFW_KEY_1);
    //Shape trans 2 (circle)
    in->addTrigger([&s, &rs](){
        if(!rs.shapeTransUIStarted || rs.selectedShape.has_value())
            return;
        rs.selectedShape = CIRCLE;
        std::cout << "Selected circle. Click a point to select a center and another point to select the radius." << std::endl;
        rs.mouseBinding->setEnabled(true);
        rs.mouseBinding->setCooldown(200ms);
    }, GLFW_KEY_2);

    return in;
}