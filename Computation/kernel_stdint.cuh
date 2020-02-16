#pragma once
#include "kernel_macros.cuh"

#ifdef BUILD_FOR_NVRTC

// stdint.h includes features.h, which includes sys/cdefs.h, which somehow breaks ntrtc completely
DEFER_TO_NVRTC_PREPROCESSOR #define _SYS_CDEFS_H  //Prevent inclusion of sys/cdefs.h
DEFER_TO_NVRTC_PREPROCESSOR #define __extension__ // For some reason ntrtc thinks __extension__ is a variable
DEFER_TO_NVRTC_PREPROCESSOR #include <stdint.h>

#else

#include <cstdint>

#endif