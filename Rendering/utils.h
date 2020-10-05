#pragma once
#include <utility>
#include "../Computation/shared_types.h"

std::pair<float, float> interleavedMinmax(const float* buffer, size_t size);
size_t deduplicateWithTol(HostFloatComplex *buffer, size_t size, float tsquare, size_t maxAttractors);
bool withinTolerance(const HostFloatComplex& a, const HostFloatComplex& b, float tsquare);