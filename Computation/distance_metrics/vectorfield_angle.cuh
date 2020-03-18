#include "../kernel_types.h"
#include "../math.cuh"
#include <math_constants.h>

__device__ __inline__ dist_t vfAngle(complex z, dist_t unused1, complex p, float unused2) {
    complex disp = csub(F(z, p), z);
    return atan2f(disp.y, disp.x) + CUDART_PI_F;
}