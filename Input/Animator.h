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

    template <typename T>
    T zoomInterpolate(double t, Interpolate<T> i) {
        static const double param = std::log(i.second / i.first);
        return i.first * std::exp(param * t);
    }
public:
    bool process([[maybe_unused]] double unused) override;

    Animator(AnimationParams params, State &state, RuntimeState &rs) : params(std::move(params)), state(state), rs(rs) {}
};



