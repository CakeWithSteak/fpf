#include "Timer.h"

Timer::Timer() {
    reset();
}

void Timer::reset() {
    t0 = std::chrono::high_resolution_clock::now();
}

std::chrono::duration<float> Timer::get() {
    return std::chrono::high_resolution_clock::now() - t0;
}

float Timer::getSeconds() {
    return get().count();
}
