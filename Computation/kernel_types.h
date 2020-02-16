#pragma once
#ifndef BUILD_FOR_NVRTC
#include <builtin_types.h>
using fpdist_t = int;
using fpdist2 = int2;
#else
#define fpdist_t int
#define fpdist2 int2
#endif

#define make_fpdist2 make_int2