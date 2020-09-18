#include "Animator.h"

bool Animator::process(double unused) {
    int totalFrames = std::ceil(params.fps * params.duration);
    if(frame >= totalFrames)
        return false;

    double t = frame / (double)totalFrames;

    state.maxIters = interpolate(t, params.maxIters);
    state.metricArg = interpolate(t, params.metricArg);
    state.p = interpolate(t, params.p);
    state.lineTransIteration = interpolate(t, params.lineTransIteration);
    state.viewport.moveTo(interpolate(t, params.viewportCenter));
    state.viewport.zoomTo(interpolate(t, params.viewportBreadth));
    if(params.colorCutoff.has_value())
        state.colorCutoff = interpolate(t, *params.colorCutoff);
    if(params.pathStart.has_value())
        rs.renderer.generatePath(interpolate(t, *params.pathStart), state.metricArg, state.p);

    if(params.lineTransStart.has_value() && params.lineTransEnd.has_value()) {
        state.lineTransStart = interpolate(t, *params.lineTransStart);
        state.lineTransEnd = interpolate(t, *params.lineTransEnd);
        rs.renderer.generateLineTransform(*state.lineTransStart, *state.lineTransEnd, state.lineTransIteration, state.p);
    }

    ++frame;
    return true;
}