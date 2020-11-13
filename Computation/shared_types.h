#pragma once

#ifndef BUILD_FOR_NVRTC
#include <complex>
#endif

//FFI types
#pragma pack(push, 1)

#ifndef BUILD_FOR_NVRTC
struct HostFloatComplex {
    float re;
    float im;

    operator std::complex<float>() const {
        return std::complex<float>(re, im);
    }
    std::complex<float> std() const {
        return static_cast<std::complex<float>>(*this);
    }
    HostFloatComplex(const std::complex<float>& z) : re(z.real()), im(z.imag()) {}
    HostFloatComplex() = default;
};

struct HostDoubleComplex {
    double re;
    double im;

    operator std::complex<double>() const {
        return std::complex<double>(re, im);
    }
    std::complex<double> std() const {
        return static_cast<std::complex<double>>(*this);
    }
    HostDoubleComplex(const std::complex<double>& z) :re(z.real()), im(z.imag()) {}
    HostDoubleComplex() = default;
};
#else
#include "kernel_macros.cuh"
RUNTIME #define HostFloatComplex float2
RUNTIME #define HostDoubleComplex double2
#endif

enum TransformShape {
    LINE,
    CIRCLE
};

struct LineProps {
    HostDoubleComplex p1;
    HostDoubleComplex p2;
};

struct CircleProps {
    HostDoubleComplex center;
    double r;
};

struct ShapeProps {
    TransformShape shape;
    union {
        LineProps line;
        CircleProps circle;
    };
};
#pragma pack(pop)
