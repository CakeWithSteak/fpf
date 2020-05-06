#pragma once
#include "kernel_macros.cuh"

#ifdef BUILD_FOR_NVRTC

// stdint.h includes features.h, which includes sys/cdefs.h, which somehow breaks nvrtc completely
RUNTIME #define _SYS_CDEFS_H  //Prevent inclusion of sys/cdefs.h
RUNTIME #define __extension__ // For some reason nvrtc thinks __extension__ is a variable
RUNTIME #include <stdint.h>

#else

#include <cstdint>

#endif