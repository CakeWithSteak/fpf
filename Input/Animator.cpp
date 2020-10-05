#include "Animator.h"

bool Animator::process(double unused) {
    if(frame >= params.totalFrames())
        return false;

    double t = frame / (double)(params.totalFrames() - 1);

    state.maxIters = interpolate(t, params.maxIters);
    state.metricArg = interpolate(t, params.metricArg);
    state.p = interpolate(t, params.p);
    state.shapeTransIteration = interpolate(t, params.lineTransIteration);
    state.viewport.moveTo(interpolate(t, params.viewportCenter));
    state.viewport.zoomTo( zoomInterpolate(t, params.viewportBreadth));
    if(params.colorCutoff.has_value())
        state.colorCutoff = interpolate(t, *params.colorCutoff);
    if(params.pathStart.has_value())
        rs.renderer.generatePath(interpolate(t, *params.pathStart), state.metricArg, state.p);

    //fixme implement animation for all shape transforms
    /*if(params.lineTransStart.has_value() && params.lineTransEnd.has_value()) {
        state.lineTransStart = interpolate(t, *params.lineTransStart);
        state.lineTransEnd = interpolate(t, *params.lineTransEnd);
        rs.renderer.generateShapeTransform(ShapeProps(), state.shapeTransIteration,
                                           state.p);
    }*/

    ++frame;
    return true;
}