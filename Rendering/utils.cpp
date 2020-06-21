#include <limits>
#include <cmath>
#include <algorithm>
#include "utils.h"

std::pair<float, float> interleavedMinmax(const float* buffer, size_t size) {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();
    for(int i = 0; i < size; i += 2) {
        if(buffer[i] < min)
            min = buffer[i];
        if(buffer[i + 1] > max)
            max = buffer[i + 1];
    }
    return {min, max};
}

size_t deduplicateWithTol(HostComplex *buffer, size_t size, float tsquare, size_t maxAttractors) {
    int j = 0;
    for(int i = 0; i < size && j < maxAttractors; ++i) {
        auto curr = buffer[i];
        if(std::isnan(curr.re))
            continue;

        bool unique = true;
        for(int k = 0; k < j; ++k) {
            if(unique = !withinTolerance(curr, buffer[k], tsquare), !unique)
                break;
        }
        if(unique) {
            buffer[j] = curr;
            ++j;
        }
    }
    //Ensures that the attractors will always show up in the same order
    std::sort(buffer, buffer + j, [](const HostComplex& a, const HostComplex& b){
        constexpr float ORDERING_TOL = 0.03f; //Account for small deviations of the attractor's position
        return (std::abs(a.re - b.re) >= ORDERING_TOL) ? (a.re < b.re) : (a.im < b.im);
    });
    return j;
}

bool withinTolerance(const HostComplex& a, const HostComplex& b, float tsquare) {
    auto xdist = a.re - b.re;
    auto ydist = a.im - b.im;
    return (xdist * xdist + ydist * ydist) <= tsquare;
}