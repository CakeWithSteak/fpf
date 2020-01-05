#pragma once
#include <complex>

using fcomplex = std::complex<float>;

class Viewport {
    fcomplex center;
    float breadth;
    void debugPrintState();
public:
    enum class Direction {
        RIGHT,
        LEFT,
        UP,
        DOWN
    };

    Viewport(const std::complex<float>& center, float breadth);

    void move(Direction dir, float step);
    void moveTo(const fcomplex& target);
    void zoom(float step);
    void zoomTo(float target);

    //Returns bottom-left and top-right corners
    [[nodiscard]] std::pair<fcomplex, fcomplex> getCorners() const;

    [[nodiscard]] const fcomplex& getCenter() const;
    [[nodiscard]] float getBreadth() const;
};


