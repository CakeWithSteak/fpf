#pragma once
#include "../utils.cuh"
#include "../constants.h"

__device__ dist_t periodic(complex z, dist_t maxIters, complex p, real tsquare, complex c) {
    __shared__ complex previouslySeen[MAX_PERIOD * MAIN_KERNEL_BLOCK_SIZE];
    const int psOffset = threadIdx.x * MAX_PERIOD;

    complex firstPeriodicP;
    int firstPeriodicIter = -1;
    for(int i = 0; i < maxIters; ++i) {
        z = F(z, p, c);
        if(i > PERIODIC_MIN_ITERATIONS) { //Iterate for a while so that only stable cycles are colored
            if (firstPeriodicIter != -1 && withinTolerance(z, firstPeriodicP, tsquare)) {
                int period = i - firstPeriodicIter;
                return (period == 1) ? -1 : period;
            }
            for (int j = 0; j < min(i, MAX_PERIOD) && firstPeriodicIter == -1; ++j) {
                if (withinTolerance(z, previouslySeen[psOffset + j], tsquare)) {
                    firstPeriodicIter = i;
                    firstPeriodicP = z;
                }
            }
        }
        previouslySeen[psOffset + (i % MAX_PERIOD)] = z;
    }
    return -1; //Not periodic
}