#pragma once
#include <../Rendering/Window.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include "Viewport.h"
#include "Timer.h"
#include "isordered.h"

using namespace std::chrono_literals;

template <typename T>
using limits = std::numeric_limits<T>;

class InputBinding {
public:
    virtual bool process(Window& window, float dt) = 0;
    virtual ~InputBinding() = default;
};

class InputHandler {
    Window& window;
    std::vector<InputBinding*> bindings;
public:
    bool process(float dt);

    template <typename T>
    void addScalar(T& val, int upKey, int downKey, T step, const std::string& displayName, T min = limits<T>::lowest(), T max = limits<T>::max());

    void addToggle(bool& val, int key, const std::string& displayName);
    void addViewport(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey, int resetKey, float moveStep, float zoomStep);

    explicit InputHandler(Window& window) : window(window) {}
    ~InputHandler();
};

template <typename T>
class ScalarBinding : public InputBinding {
    T& val;
    T step, min, max;
    int upKey, downKey;
    std::string displayName;
    bool isUncomparable;
public:
    virtual bool process(Window& window, float dt) override;
    ScalarBinding(T& val, int upKey, int downKey, T step, std::string displayName, T min = limits<T>::lowest(), T max = limits<T>::max()) :
        val(val), min(min), max(max), upKey(upKey), downKey(downKey), step(step), displayName(displayName) {}
};

class ToggleBinding : public InputBinding {
    constexpr static auto cooldown = 400ms;
    Timer timer;
    bool& val;
    int key;
    std::string displayName;
public:
    virtual bool process(Window& window, float dt) override;
    ToggleBinding(bool& val, int key, std::string displayName) :
        val(val), key(key), displayName(displayName) {}
};

class ViewportBinding : public InputBinding {
    Viewport& v;
    int upKey, downKey, leftKey, rightKey, zoomInKey, zoomOutKey, resetKey;
    float moveStep, zoomStep;
    std::complex<float> resetPoint;
    float resetZoom;
public:
    virtual bool process(Window& window, float dt) override;
    ViewportBinding(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey,
        int resetKey, float moveStep, float zoomStep) : v(v), upKey(upKey),
        downKey(downKey), leftKey(leftKey), rightKey(rightKey), zoomInKey(zoomInKey), zoomOutKey(zoomOutKey),
        resetKey(resetKey), resetPoint(v.getCenter()), resetZoom(v.getBreadth()), moveStep(moveStep), zoomStep(zoomStep) {}
};

template<typename T>
void InputHandler::addScalar(T& val, int upKey, int downKey, T step, const std::string& displayName, T min, T max) {
    bindings.push_back(new ScalarBinding(val, upKey, downKey, step, displayName, min, max));
}

template<typename T>
bool ScalarBinding<T>::process(Window& window, float dt) {
    if constexpr(std::is_integral_v<T>)
        dt = 1;

    bool inputReceived = false;
    if(window.isKeyPressed(upKey)) {
        val += step * dt;
        inputReceived = true;
    }
    if(window.isKeyPressed(downKey)) {
        val -= step * dt;
        inputReceived = true;
    }

    if(inputReceived) {
        if constexpr(is_ordered_v<T>)
            val = std::clamp(val, min, max);
        std::cout << displayName << ": " << val << std::endl;
    }

    return inputReceived;
}