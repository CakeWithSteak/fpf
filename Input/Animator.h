#pragma once
#include "Controller.h"
#include "AnimationParams.h"
#include "../utils/State.h"

class Animator : public Controller {
    const AnimationParams params;
    State& state;
    RuntimeState& rs;
    int frame = 0;

    template <typename T>
    T interpolate(double t, Interpolate<T> i) {
        return i.first + t * (i.second - i.first);
    }

    ShapeProps interpolate(double t, Interpolate<ShapeProps> i) {
        assert(i.first.shape == i.second.shape);
        switch(i.first.shape) {
            case LINE:
                return { .shape = LINE,
                        .line = { .p1 = interpolate(t, Interpolate<std::complex<double>>{i.first.line.p1.std(), i.second.line.p1.std()}),
                                  .p2 = interpolate(t, Interpolate<std::complex<double>>{i.first.line.p2.std(), i.second.line.p2.std()}) }};
            case CIRCLE:
                return { .shape = CIRCLE,
                         .circle = { .center =  interpolate(t, Interpolate<std::complex<double>>{i.first.circle.center.std(), i.second.circle.center.std()}),
                                     .r = interpolate(t, Interpolate<double>{i.first.circle.r, i.second.circle.r}) }};
            default:
                throw std::runtime_error("Invalid tag on ShapeProps");
        }
    }

    template <typename T>
    T zoomInterpolate(double t, Interpolate<T> i) {
        static const double param = std::log(i.second / i.first);
        return i.first * std::exp(param * t);
    }
public:
    bool process([[maybe_unused]] double unused) override;

    Animator(AnimationParams params, State &state, RuntimeState &rs) : params(std::move(params)), state(state), rs(rs) {}
};



