#pragma once
#include <complex>
#include <boost/serialization/access.hpp>

using dcomplex = std::complex<double>;

class Viewport {
    dcomplex center;
    double breadth;
    void debugPrintState();
public:
    enum class Direction {
        RIGHT,
        LEFT,
        UP,
        DOWN
    };

    Viewport(const dcomplex& center, double breadth);
    Viewport() = default;

    void move(Direction dir, double step);
    void moveTo(const dcomplex& target);
    void zoom(double step);
    void zoomTo(double target);

    //Returns bottom-left and top-right corners
    [[nodiscard]] std::pair<dcomplex, dcomplex> getCorners() const;

    [[nodiscard]] const dcomplex& getCenter() const;
    [[nodiscard]] double getBreadth() const;
    dcomplex resolveScreenCoords(double x, double y, double width, double height) const;

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & center;
        ar & breadth;
    }
};


