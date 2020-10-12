#pragma once
#include <string>
#include <complex>
#include "Viewport.h"
#include "../Rendering/Window.h"
#include "../Rendering/Renderer.h"
#include "../Input/Input.h"
#include "../cli.h"
#include "../modes.h"
#include "AnimationExporter.h"

//A struct encompassing program state useful for event handlers and serialization
struct State {
    std::string expr;
    int maxIters;
    double metricArg;
    std::complex<double> p;
    Viewport viewport;
    bool colorCutoffEnabled;
    double colorCutoff;
    int width, height;
    ModeInfo mode; //Only the DistanceMetric is serialized
    std::optional<std::complex<double>> pathStart;
    std::optional<ShapeProps> shapeTransProps;
    int shapeTransIteration = 0;
    bool forceDisableIncrementalShapeTrans = false;
    bool doublePrec = false;
    int shapeTransNumPointsOverride = -1;

    explicit State(const Options& opt) {
        expr = opt.expression;
        mode = opt.mode;
        width = opt.width;
        height = opt.height;
        metricArg = opt.metricArg;
        forceDisableIncrementalShapeTrans = opt.forceDisableIncrementalShapeTrans;
        colorCutoffEnabled = opt.colorCutoff.has_value();
        colorCutoff = colorCutoffEnabled ? *opt.colorCutoff : 10.0f;
        maxIters = opt.maxIters;
        doublePrec = opt.doublePrec;
        p = opt.p;
        viewport = Viewport(opt.viewportCenter, opt.viewportBreadth);
    }
    State() = default;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & expr;
        ar & maxIters;
        ar & metricArg;
        ar & p;
        ar & viewport;
        ar & colorCutoffEnabled;
        ar & colorCutoff;
        ar & width;
        ar & height;
        ar & mode;
        ar & pathStart;
        ar & shapeTransIteration;
        ar & shapeTransProps;
        ar & forceDisableIncrementalShapeTrans;
        ar & shapeTransNumPointsOverride;
    }
};

struct RuntimeState {
    Window& window;
    Renderer& renderer;
    bool forceRerender = false;
    std::optional<std::filesystem::path> refsPath;
    InputBinding* mouseBinding;
    std::string animExportBasename;
    std::optional<AnimationExporter> animExport;

    //State for the UI
    bool shapeTransUIStarted = false;
    bool shapeTransUIFinished = false;
    std::optional<TransformShape> selectedShape;
    std::optional<std::complex<double>> lineTransStart;
    std::optional<std::complex<double>> circleCenter;
};