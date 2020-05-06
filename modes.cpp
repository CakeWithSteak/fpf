#include "modes.h"
#include <cmath>

constexpr float TAU = 3.14159265358979323846 * 2;

const std::map<DistanceMetric, ModeInfo> modes {
        {FIXEDPOINT_ITERATIONS, {
            .metric = FIXEDPOINT_ITERATIONS,
            .displayName = "Fixed point distance (iterations)",
            .cliName = "fixed",
            .metricInternalName = "FIXEDPOINT_ITERATIONS",
            .argDisplayName = "Tolerance",
            .argInitValue = 0.01f,
            .argStep = 0.0025f,
            .argMin = 0.0f,
            .argMax = 2.0f,
            .argIsTolerance = true
        }},
        {JULIA, {
            .metric = JULIA,
            .displayName = "Julia set",
            .cliName = "julia",
            .metricInternalName = "JULIA",
            .argDisplayName = "Escape radius",
            .argInitValue = 10.0f,
            .argStep = 0.5f,
            .argMin = 0.0f,
            .argMax = std::numeric_limits<float>::max(),
        }},
        {FIXEDPOINT_EUCLIDEAN, {
           .metric = FIXEDPOINT_EUCLIDEAN,
           .displayName = "Fixed point distance (Euclidean)",
           .cliName = "fixed-dist",
           .metricInternalName = "FIXEDPOINT_EUCLIDEAN",
           .argDisplayName = "Tolerance",
           .argInitValue = 0.01f,
           .argStep = 0.0025f,
           .argMin = 0.0f,
           .argMax = 2.0f,
           .argIsTolerance = true
        }},
        {VECTORFIELD_MAGNITUDE, {
           .metric = VECTORFIELD_MAGNITUDE,
           .displayName = "Displacement",
           .cliName = "displacement",
           .metricInternalName = "VECTORFIELD_MAGNITUDE",
           .argDisplayName = "",
           .argInitValue = 0.0f,
           .argStep = 0.0f,
           .argMin = 0.0f,
           .argMax = 0.0f,
           .argIsTolerance = false,
           .defaultColorCutoff = 20.0f,
           .disableArg = true,
           .disableIterations = true
        }},
        {VECTORFIELD_ANGLE, {
            .metric = VECTORFIELD_ANGLE,
            .displayName = "Direction",
            .cliName = "direction",
            .metricInternalName = "VECTORFIELD_ANGLE",
            .argDisplayName = "",
            .argInitValue = 0.0f,
            .argStep = 0.0f,
            .argMin = 0.0f,
            .argMax = 0.0f,
            .argIsTolerance = false,
            .defaultColorCutoff = -1.0f,
            .disableArg = true,
            .disableIterations = true,
            .maxHue = 1.0f, // The point of setting a max hue is to prevent high values bleeding into small ones -
                            // but this is exactly what we want with angles (eg 1° and 359° should look similar)
            .staticMinMax = {{0, TAU}}
        }},
        {CAPTURING_JULIA, {
            .metric = JULIA, //fixme This will not serialize properly
            .displayName = "Julia set with capture",
            .cliName = "julia-capt",
            .metricInternalName = "JULIA",
            .argDisplayName = "Escape radius",
            .argInitValue = 2.0f,
            .argStep = 0.2f,
            .argMin = 0.0f,
            .argMax = std::numeric_limits<float>::max(),
            .capturing = true,
            .disablePath = true,
            .initMaxIters = 512
        }},
        {CAPTURING_FIXEDPOINT, {
          .metric = FIXEDPOINT_ITERATIONS,
          .displayName = "Fixed point with capture",
          .cliName = "fixed-capt",
          .metricInternalName = "FIXEDPOINT_ITERATIONS",
          .argDisplayName = "Tolerance",
          .argInitValue = 0.00035f,
          .argStep = 0.00025f,
          .argMin = 0.0f,
          .argMax = 2.0f,
          .capturing = true,
          .disablePath = true,
          .initMaxIters = 512
        }},
};