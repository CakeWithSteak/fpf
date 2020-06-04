#pragma once

#include "../Computation/kernel_types.h"
#include <utility>

#pragma pack(push, 1)
struct HostComplex {
    float re;
    float im;
};
#pragma pack(pop)

std::pair<dist_t, dist_t> interleavedMinmax(const dist_t* buffer, size_t size);
size_t deduplicateWithTol(HostComplex* buffer, size_t size, float tsquare);
bool withinTolerance(const HostComplex& a, const HostComplex& b, float tsquare);