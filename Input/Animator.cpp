#include "Animator.h"

bool Animator::process(double unused) {
    if(frame >= params.totalFrames())
        return false;

    double t = frame / (double)(params.totalFrames() - 1);

    state.maxIters = interpolate(t, params.maxIters);
    state.metricArg = interpolate(t, params.metricArg);
    state.p = interpolate(t, params.p);
    state.shapeTransIteration = interpolate(t, params.shapeTransIteration);
    state.viewport.moveTo(interpolate(t, params.viewportCenter));
    state.viewport.zoomTo( zoomInterpolate(t, params.viewportBreadth));
    if(params.colorCutoff.has_value())
        state.colorCutoff = interpolate(t, *params.colorCutoff);
    if(params.pathStart.has_value())
        rs.renderer.generatePath(interpolate(t, *params.pathStart), state.metricArg, state.p);
    if(params.shapeProps.has_value()) {
        state.shapeTransProps = interpolate(t, *params.shapeProps);
        state.shapeTransIteration = interpolate(t, params.shapeTransIteration);
        rs.renderer.generateShapeTransform(*state.shapeTransProps, state.shapeTransIteration, state.p, 0);
    }

    ++frame;
    return true;
}