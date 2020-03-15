#pragma once
#ifndef BUILD_FOR_NVRTC
#include <builtin_types.h>
#endif

using dist_t = float;
using dist2 = float2;
#define make_dist2 make_float2
using complex = float2;

#ifdef BUILD_FOR_NVRTC
__device__ __inline__ complex F(complex z, complex p); //Forward declaration for generated code
#endif