#pragma once

#ifndef BUILD_FOR_NVRTC //Hack for code insights
#include <vector_functions.h>
#include <cmath>
#endif


RUNTIME #ifdef PREC_FLOAT
#include "math_float.cuh"
RUNTIME #elif defined(PREC_DOUBLE)
#include "math_double.cuh"
RUNTIME #endif