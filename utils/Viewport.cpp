#include <iostream>
#include "Viewport.h"

Viewport::Viewport(const std::complex<float>& center, float breadth)
        : center(center), breadth(breadth) {}

const fcomplex& Viewport::getCenter() const {
    return center;
}

float Viewport::getBreadth() const {
    return breadth;
}

void Viewport::move(Viewport::Direction dir, float step) {
    switch(dir) {
        case Direction::UP:
            center.imag(center.imag() + breadth * step);
            break;
        case Direction::DOWN:
            center.imag(center.imag() - breadth * step);
            break;
        case Direction::RIGHT:
            center += breadth * step;
            break;
        case Direction::LEFT:
            center -= breadth * step;
            break;
    }
    debugPrintState();
}

void Viewport::moveTo(const fcomplex& target) {
    center = target;
    debugPrintState();
}

void Viewport::zoom(float step) {
    breadth -= step * breadth;
    debugPrintState();
}

void Viewport::zoomTo(float target) {
    breadth = target;
    debugPrintState();
}

std::pair<fcomplex, fcomplex> Viewport::getCorners() const {
    auto bottomLeft = fcomplex(center.real() - breadth, center.imag() - breadth);
    auto topRight = fcomplex(center.real() + breadth, center.imag() + breadth);
    return {bottomLeft, topRight};
}

inline void Viewport::debugPrintState() {
    /**/
    auto [bottom, top] = getCorners();
    std::cout << bottom << " -> " << top << std::endl;
    /**/
}

fcomplex Viewport::resolveScreenCoords(double x, double y, double width, double height) const {
    float re = breadth * (2 * (x / width) - 1) + center.real();
    float im = breadth * (1 - 2*y / height) + center.imag();
    return {re, im};
}
