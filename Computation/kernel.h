#pragma once
#include <builtin_types.h>

using fpdist_t = int;

void launch_kernel(float re0, float re1, float im0, float im1, float tolerance, fpdist_t maxIters, fpdist_t* minmaxOut, cudaSurfaceObject_t surface, int surfW, int surfH);