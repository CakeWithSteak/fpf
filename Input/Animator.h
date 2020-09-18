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
public:
    bool process([[maybe_unused]] double unused) override;

    Animator(AnimationParams params, State &state, RuntimeState &rs) : params(std::move(params)), state(state), rs(rs) {}
};



