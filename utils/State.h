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
    std::optional<std::complex<double>> lineTransStart;
    std::optional<std::complex<double>> lineTransEnd;
    int lineTransIteration = 0;
    bool lineTransEnabled = false; // Never serialised, inferred from lineTransEnd during deserialization
    bool forceDisableIncrementalLineTracing = false;
    bool doublePrec = false;

    explicit State(const Options& opt) {
        expr = opt.expression;
        mode = opt.mode;
        width = opt.width;
        height = opt.height;
        metricArg = opt.metricArg;
        forceDisableIncrementalLineTracing = opt.forceDisableIncrementalLineTracing;
        colorCutoffEnabled = opt.colorCutoff.has_value();
        colorCutoff = colorCutoffEnabled ? *opt.colorCutoff : 10.0f;
        maxIters = opt.maxIters;
        doublePrec = opt.doublePrec;
        p = opt.p;
        viewport = Viewport(opt.viewportCenter, opt.viewportBreadth);
    }
    State() = default;

    template<class Archive>
    void save(Archive& ar, const unsigned int version) const
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
        ar & lineTransIteration;
        if(lineTransEnd.has_value()) { // Only serialize line trans mode if both the start and the end of the line are given, otherwise things will probably break
            ar & lineTransStart;
            ar & lineTransEnd;
        } else {
            ar & std::optional<std::complex<double>>();
            ar & std::optional<std::complex<double>>();
        }
        ar & forceDisableIncrementalLineTracing;
    }

    template<class Archive>
    void load(Archive & ar, const unsigned int version)
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
        ar & lineTransIteration;
        ar & lineTransStart;
        ar & lineTransEnd;
        lineTransEnabled = lineTransEnd.has_value();
        ar & forceDisableIncrementalLineTracing;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

struct RuntimeState {
    Window& window;
    Renderer& renderer;
    bool forceRerender = false;
    std::optional<std::filesystem::path> refsPath;
    InputBinding* mouseBinding;
    std::string animExportBasename;
    std::optional<AnimationExporter> animExport;
};