#pragma once
#include <utility>

#pragma pack(push, 1)
struct HostComplex {
    float re;
    float im;
};
#pragma pack(pop)

std::pair<float, float> interleavedMinmax(const float* buffer, size_t size);
size_t deduplicateWithTol(HostComplex *buffer, size_t size, float tsquare, size_t maxAttractors);
bool withinTolerance(const HostComplex& a, const HostComplex& b, float tsquare);