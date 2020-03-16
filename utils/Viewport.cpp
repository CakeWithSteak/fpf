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
    if(breadth < 1)
        step *= breadth;
    switch(dir) {
        case Direction::UP:
            center.imag(center.imag() + step);
            break;
        case Direction::DOWN:
            center.imag(center.imag() - step);
            break;
        case Direction::RIGHT:
            center += step;
            break;
        case Direction::LEFT:
            center -= step;
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