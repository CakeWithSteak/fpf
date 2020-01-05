#pragma once
#include <chrono>

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;
public:
    Timer();
    void reset();
    std::chrono::duration<float> get();
    float getSeconds();
};


