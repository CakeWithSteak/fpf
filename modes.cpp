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
            .argInitValue = 0.01,
            .argStep = 0.0025,
            .argMin = 0.0,
            .argMax = 2.0,
            .argIsTolerance = true
        }},
        {JULIA, {
            .metric = JULIA,
            .displayName = "Julia set",
            .cliName = "julia",
            .metricInternalName = "JULIA",
            .argDisplayName = "Escape radius",
            .argInitValue = 10.0,
            .argStep = 0.5,
            .argMin = 0.0,
            .argMax = std::numeric_limits<double>::max(),
        }},
        {FIXEDPOINT_EUCLIDEAN, {
           .metric = FIXEDPOINT_EUCLIDEAN,
           .displayName = "Fixed point distance (Euclidean)",
           .cliName = "fixed-dist",
           .metricInternalName = "FIXEDPOINT_EUCLIDEAN",
           .argDisplayName = "Tolerance",
           .argInitValue = 0.01,
           .argStep = 0.0025,
           .argMin = 0.0,
           .argMax = 2.0,
           .argIsTolerance = true
        }},
        {VECTORFIELD_MAGNITUDE, {
           .metric = VECTORFIELD_MAGNITUDE,
           .displayName = "Displacement",
           .cliName = "displacement",
           .metricInternalName = "VECTORFIELD_MAGNITUDE",
           .argDisplayName = "",
           .argInitValue = 0.0,
           .argStep = 0.0,
           .argMin = 0.0,
           .argMax = 0.0,
           .argIsTolerance = false,
           .defaultColorCutoff = 20.0,
           .disableArg = true,
           .disableIterations = true
        }},
        {VECTORFIELD_ANGLE, {
            .metric = VECTORFIELD_ANGLE,
            .displayName = "Direction",
            .cliName = "direction",
            .metricInternalName = "VECTORFIELD_ANGLE",
            .argDisplayName = "",
            .argInitValue = 0.0,
            .argStep = 0.0,
            .argMin = 0.0,
            .argMax = 0.0,
            .argIsTolerance = false,
            .defaultColorCutoff = -1.0,
            .disableArg = true,
            .disableIterations = true,
            .maxHue = 1.0,  // The point of setting a max hue is to prevent high values bleeding into small ones -
                            // but this is exactly what we want with angles (eg 1° and 359° should look similar)
            .staticMinMax = {{0, TAU}}
        }},
        {CAPTURING_JULIA, {
            .metric = JULIA,
            .serializedName = CAPTURING_JULIA,
            .displayName = "Julia set with capture",
            .cliName = "julia-capt",
            .metricInternalName = "JULIA",
            .argDisplayName = "Escape radius",
            .argInitValue = 2.0,
            .argStep = 0.2,
            .argMin = 0.0,
            .argMax = std::numeric_limits<double>::max(),
            .capturing = true,
            .initMaxIters = 512
        }},
        {CAPTURING_FIXEDPOINT, {
          .metric = FIXEDPOINT_ITERATIONS,
          .serializedName = CAPTURING_FIXEDPOINT,
          .displayName = "Fixed point with capture",
          .cliName = "fixed-capt",
          .metricInternalName = "FIXEDPOINT_ITERATIONS",
          .argDisplayName = "Tolerance",
          .argInitValue = 0.00035,
          .argStep = 0.00025,
          .argMin = 0.0,
          .argMax = 2.0,
          .capturing = true,
          .initMaxIters = 512
        }},
        {ATTRACTOR, {
           .metric = ATTRACTOR,
           .displayName = "By attractor",
           .cliName = "attractor",
           .metricInternalName = "ATTRACTOR",
           .argDisplayName = "Tolerance",
           .argInitValue = 0.0005,
           .argStep = 0.0025,
           .argMin = 0.0,
           .argMax = 2.0,
           .argIsTolerance = true,
           .initMaxIters = 512,
           .isAttractor = true,
        }},
};