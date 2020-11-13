#include <iostream>
#include <cassert>
#include "Viewport.h"
#include <algorithm>

Viewport::Viewport(const dcomplex& center, double breadth)
        : center(center), breadth(breadth) {}

const dcomplex& Viewport::getCenter() const {
    return center;
}

double Viewport::getBreadth() const {
    return breadth;
}

void Viewport::move(Viewport::Direction dir, double step) {
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

void Viewport::moveTo(const dcomplex& target) {
    center = target;
    debugPrintState();
}

void Viewport::zoom(double step) {
    breadth -= step * breadth;
    breadth = std::max(0.0, breadth);
    debugPrintState();
}

void Viewport::zoomTo(double target) {
    assert(target >= 0.0);
    breadth = target;
    debugPrintState();
}

std::pair<dcomplex, dcomplex> Viewport::getCorners() const {
    auto bottomLeft = dcomplex(center.real() - breadth, center.imag() - breadth);
    auto topRight = dcomplex(center.real() + breadth, center.imag() + breadth);
    return {bottomLeft, topRight};
}

inline void Viewport::debugPrintState() {
    /**/
    auto [bottom, top] = getCorners();
    std::cout << bottom << " -> " << top << std::endl;
    /**/
}

dcomplex Viewport::resolveScreenCoords(double x, double y, double width, double height) const {
    double re = breadth * (2 * (x / width) - 1) + center.real();
    double im = breadth * (1 - 2*y / height) + center.imag();
    return {re, im};
}
