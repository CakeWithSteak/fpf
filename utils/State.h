#pragma once
#include <string>
#include <complex>
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
    double metricArg;
    std::complex<double> p{0};
    Viewport viewport;
    bool colorCutoffEnabled;
    double colorCutoff;
    int width, height;
    ModeInfo mode; //Only the DistanceMetric is serialized
    std::optional<std::complex<double>> pathStart = {};
    std::optional<std::complex<double>> lineTransStart = {};
    std::optional<std::complex<double>> lineTransEnd = {};
    int lineTransIteration = 0;
    bool lineTransEnabled = false; // Never serialised, inferred from lineTransEnd during deserialization
    bool forceDisableIncrementalLineTracing = false;

    explicit State(const Options& opt) {
        expr = opt.expression;
        mode = opt.mode;
        width = opt.width;
        height = opt.height;
        metricArg = (opt.metricArg.has_value()) ? opt.metricArg.value() : mode.argInitValue;
        forceDisableIncrementalLineTracing = opt.forceDisableIncrementalLineTracing;
        colorCutoffEnabled = (mode.defaultColorCutoff != -1);
        colorCutoff = colorCutoffEnabled ? mode.defaultColorCutoff : 10.0f;
        maxIters = mode.initMaxIters;
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
    std::filesystem::path refsPath;
    InputBinding* mouseBinding;
};