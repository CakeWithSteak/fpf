#pragma once
#include "Input/Input.h"
#include "utils/State.h"
#include "utils/serialization.h"
#include "utils/imageExport.h"
#include <optional>

struct ShapeTransformInfo {
    TransformShape shape;
    std::string name;
    std::string instruction1;
    bool setNumPoints = false;
    std::optional<std::string> instruction2 = {};
};

extern const std::vector<ShapeTransformInfo> shapeTransformUIInfo;

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
    in->addTrigger([&rs]() {rs.window.setShouldClose(true);}, GLFW_KEY_ESCAPE);
    in->addTrigger([&s, &rs](){rs.window.minimize(); save(s); rs.window.restore();}, GLFW_KEY_TAB);

    if(!mode.disableOverlays) {
        rs.mouseBinding = &in->addTrigger([&s, &rs](double x, double y) {
            auto z = s.viewport.resolveScreenCoords(x, y, rs.window.getWidth(), rs.window.getHeight());
            if(!rs.shapeTransUIStarted && !rs.shapeTransUIFinished) { //Path mode
                s.pathStart = z;
                std::cout << "Rendering path overlay from " << z << ".\n";
                auto length = rs.renderer.generatePath(z, s.metricArg, s.p);
                std::cout << "Path length: " << length << std::endl;
                return true;
            } else if(rs.selectedShape.has_value() && rs.shapeTransUIStarted){ // Shape transform mode
                if(rs.selectedShape == LINE) {
                    if (!rs.lineTransStart.has_value()) { //First click to select start point
                        rs.lineTransStart = z;
                        std::cout << "Set line transform start point to " << *rs.lineTransStart << "." << std::endl;
                        return false;
                    } else { //Second click to select endpoint
                        s.shapeTransProps = ShapeProps{LINE, {.line = {.p1 = *rs.lineTransStart, .p2 = z}}};
                        rs.renderer.generateShapeTransform(*s.shapeTransProps, s.shapeTransIteration, s.p);
                        std::cout << "Rendering line transform from " << *rs.lineTransStart << " to " << z
                                  << "." << std::endl;
                        rs.lineTransStart = {}; // Unset no longer needed state here to avoid having to do it somewhere else later
                        rs.shapeTransUIStarted = false;
                        rs.shapeTransUIFinished = true;
                        return true;
                    }
                } else if(rs.selectedShape == CIRCLE) {
                    if(!rs.circleCenter.has_value()) {
                        rs.circleCenter = z;
                        std::cout << "Set circle center to " << z << std::endl;
                        return false;
                    } else {
                        double r = std::abs(z - *rs.circleCenter);
                        s.shapeTransProps = ShapeProps{CIRCLE, {.circle = {*rs.circleCenter, r}}};
                        rs.renderer.generateShapeTransform(*s.shapeTransProps, s.shapeTransIteration, s.p, s.shapeTransNumPointsOverride);
                        std::cout << "Rendering circle transform from " << *rs.circleCenter << ", r = " << r
                                  << "." << std::endl;
                        rs.circleCenter = {}; // Unset no longer needed state here to avoid having to do it somewhere else later
                        rs.shapeTransUIStarted = false;
                        rs.shapeTransUIFinished = true;
                        return true;
                    }
                }
            }
            return false;
        }, GLFW_MOUSE_BUTTON_1);

        in->addTrigger([&s, &rs]() {
            bool overlayActive;
            if((overlayActive = rs.renderer.isOverlayActive()))
                rs.renderer.hideOverlay();
            s.pathStart = {};
            rs.shapeTransUIStarted = false; rs.shapeTransUIFinished = false; rs.selectedShape = {}; s.shapeTransNumPointsOverride = -1;
            rs.mouseBinding->setCooldown(0ms); // If we were in line trans mode we need to unset the cooldown
            return overlayActive;
        }, GLFW_KEY_H);

        in->addTrigger([&s, &rs]() {
            if(rs.shapeTransUIStarted || rs.shapeTransUIFinished)
                 return false;
            bool overlayActive;
            if((overlayActive = rs.renderer.isOverlayActive()))
                rs.renderer.hideOverlay();
            rs.shapeTransUIStarted = true;
            s.shapeTransIteration = 0;
            std::cout << "Shape transform mode enabled. Select a shape:\n";
            for(int i = 0; i < shapeTransformUIInfo.size(); ++i) {
                std::cout << "\t" << i + 1 << ": " << shapeTransformUIInfo[i].name << std::endl;
            }
            rs.mouseBinding->setEnabled(false);
            return overlayActive;
        }, GLFW_KEY_T);

        in->addTrigger([&s, &rs]() {
            if(rs.shapeTransUIFinished) {
                ++s.shapeTransIteration;
                rs.renderer.setShapeTransformIteration(s.shapeTransIteration, s.p, s.forceDisableIncrementalShapeTrans);
                std::cout << "Shape transform iteration: " << s.shapeTransIteration << "." << std::endl;
                return true;
            }
            return false;
        }, GLFW_KEY_RIGHT_SHIFT).setCooldown(150ms);

        in->addTrigger([&s, &rs]() {
            if(rs.shapeTransUIFinished && s.shapeTransIteration != 0) {
                --s.shapeTransIteration;
                rs.renderer.setShapeTransformIteration(s.shapeTransIteration, s.p, s.forceDisableIncrementalShapeTrans);
                std::cout << "Shape transform iteration: " << s.shapeTransIteration << "." << std::endl;
                return true;
            }
            return false;
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
        return false;
    }, GLFW_KEY_INSERT);

    for(int i = 0; i < shapeTransformUIInfo.size(); ++i) {
        in->addTrigger([&s, &rs, i](){
            if(!rs.shapeTransUIStarted || rs.selectedShape.has_value())
                return false;
            const auto uiInfo = shapeTransformUIInfo[i];
            rs.selectedShape = uiInfo.shape;
            std::cout << "Selected " << uiInfo.name << ". " << uiInfo.instruction1 << std::endl;
            if(uiInfo.setNumPoints) {
                rs.window.minimize();
                int num = -1;
                while(num <= 2) {
                    std::cout << "> ";
                    if(std::cin.fail()) {
                        std::cin.clear();
                        std::cin.ignore(30000, '\n');
                    }
                    std::cin >> num;
                }
                s.shapeTransNumPointsOverride = num + 1; //We draw one more point so as to close the shape
                if(uiInfo.instruction2.has_value())
                    std::cout << *uiInfo.instruction2 << std::endl;
                rs.window.restore();
            }
            rs.mouseBinding->setEnabled(true);
            rs.mouseBinding->setCooldown(200ms);
            return false;
        }, GLFW_KEY_1 + i);
    }
    return in;
}

const std::vector<ShapeTransformInfo> shapeTransformUIInfo {
        {
            .shape = LINE,
            .name = "line",
            .instruction1 = "Click two points to draw a line."
        },
        {
            .shape = CIRCLE,
            .name = "circle",
            .instruction1 = "Click a point to select a center and another point to select the radius."
        },
        {
            .shape = CIRCLE,
            .name = "polygon",
            .instruction1 = "Enter the number of points.",
            .setNumPoints = true,
            .instruction2 = "Click a point to select a center and another point to select the radius."   
        }
};