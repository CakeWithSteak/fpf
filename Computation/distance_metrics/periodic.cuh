#pragma once
#include "../kernel_types.h"
#include "../utils.cuh"
#include "../constants.h"

__device__ dist_t periodicType(complex z, dist_t maxIters, complex p, float tsquare, complex c) {
    __shared__ complex previouslySeen[MAX_PERIOD * MAIN_KERNEL_BLOCK_SIZE];
    const complex first = z;
    const int psOffset = threadIdx.x * MAX_PERIOD;

    //float2 last = z;
    for(int i = 0; i < maxIters; ++i) {
        z = F(z, p, c);
        if(withinTolerance(z, first, 2 * tsquare))
            return 2; //Periodic
        for(int j = 0; j < min(i, MAX_PERIOD); ++j) {
            if(withinTolerance(z, previouslySeen[psOffset + j], tsquare))
                return 1; //Preperiodic
        }
        previouslySeen[psOffset + (i % MAX_PERIOD)] = z;
        /*if(withinTolerance(z, last, tsquare))
            return i + 1;*/
        //last = z;
    }
    return -1; //Not periodic
}