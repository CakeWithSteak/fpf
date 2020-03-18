#include "modes.h"

const std::map<DistanceMetric, ModeInfo> modes{
        {FIXEDPOINT_ITERATIONS, {
            .metric = FIXEDPOINT_ITERATIONS,
            .displayName = "Fixed point distance (iterations)",
            .cliName = "fixed",
            .internalName = "FIXEDPOINT_ITERATIONS",
            .argDisplayName = "Tolerance",
            .argInitValue = 0.01f,
            .argStep = 0.0025f,
            .argMin = 0.0f,
            .argMax = 2.0f,

        }},
        {JULIA, {
            .metric = JULIA,
            .displayName = "Julia set",
            .cliName = "julia",
            .internalName = "JULIA",
            .argDisplayName = "Escape radius",
            .argInitValue = 10.0f,
            .argStep = 0.5f,
            .argMin = 0.0f,
            .argMax = 200.0f,

        }},
        {FIXEDPOINT_EUCLIDEAN, {
           .metric = FIXEDPOINT_EUCLIDEAN,
           .displayName = "Fixed point distance (Euclidean)",
           .cliName = "fixed-dist",
           .internalName = "FIXEDPOINT_EUCLIDEAN",
           .argDisplayName = "Tolerance",
           .argInitValue = 0.01f,
           .argStep = 0.0025f,
           .argMin = 0.0f,
           .argMax = 2.0f,
        }},
        {VECTORFIELD_MAGNITUDE, {
           .metric = VECTORFIELD_MAGNITUDE,
           .displayName = "Displacement",
           .cliName = "displacement",
           .internalName = "VECTORFIELD_MAGNITUDE",
           .argDisplayName = "",
           .argInitValue = 0.0f,
           .argStep = 0.0f,
           .argMin = 0.0f,
           .argMax = 0.0f,
           .defaultColorCutoff = 20.0f,
           .disableArg = true,
           .disableIterations = true
        }},
        {VECTORFIELD_ANGLE, {
            .metric = VECTORFIELD_ANGLE,
            .displayName = "Direction",
            .cliName = "direction",
            .internalName = "VECTORFIELD_ANGLE",
            .argDisplayName = "",
            .argInitValue = 0.0f,
            .argStep = 0.0f,
            .argMin = 0.0f,
            .argMax = 0.0f,
            .defaultColorCutoff = -1.0f,
            .disableArg = true,
            .disableIterations = true,
            .maxHue = 1.0f // The point of setting a max hue is to prevent high values bleeding into small ones -
                           // but this is exactly what we want with angles (eg 1° and 359° should look similar)
        }},
};