#include "../kernel_types.cuh"
#include "../math.cuh"
#include <math_constants.h>

__device__ __inline__ dist_t vfAngle(complex z, dist_t unused1, complex p, real unused2, complex c) {
    complex disp = csub(F(z, p, c), z);
    real angle = atan2(disp.y, disp.x);
    if(angle < 0)
        angle += 2 * CUDART_PI;
    return angle;
}