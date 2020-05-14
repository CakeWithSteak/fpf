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
protected:
    Timer timer;
    std::chrono::milliseconds cooldown = 0ms;
public:
    virtual bool process(Window& window, float dt, float multiplier) = 0;
    virtual ~InputBinding() = default;
    virtual void setCooldown(const std::chrono::milliseconds& val) {
        cooldown = val;
    }
};

class InputHandler {
    Window& window;
    std::vector<InputBinding*> bindings;
    std::optional<int> multiplierKey;
    float multiplier = 1.f;
public:
    bool process(float dt);

    template <typename T>
    InputBinding& addScalar(T& val, int upKey, int downKey, T step, const std::string& displayName, T min = limits<T>::lowest(), T max = limits<T>::max());

    InputBinding& addToggle(bool& val, int key, const std::string& displayName);
    InputBinding& addViewport(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey, int resetKey, float moveStep, float zoomStep);
    InputBinding& addTrigger(const std::function<void()>& handler, int triggerKey);
    InputBinding& addTrigger(const std::function<void(double, double)>& handler, int triggerButton);

    InputHandler(Window& window, const std::optional<int>& multiplierKey = {}, float multiplier = 1.f) : window(window),
                                                                                                         multiplierKey(multiplierKey),
                                                                                                         multiplier(multiplier) {}
    ~InputHandler();
};

template <typename T>
class ScalarBinding : public InputBinding {
    T& val;
    T step, min, max;
    int upKey, downKey;
    std::string displayName;
public:
    virtual bool process(Window& window, float dt, float multiplier) override;
    ScalarBinding(T& val, int upKey, int downKey, T step, std::string displayName, T min = limits<T>::lowest(), T max = limits<T>::max()) :
        val(val), min(min), max(max), upKey(upKey), downKey(downKey), step(step), displayName(displayName) {}
};

class ToggleBinding : public InputBinding {
    bool& val;
    int key;
    std::string displayName;
public:
    virtual bool process(Window& window, float dt, float multiplier) override;
    ToggleBinding(bool& val, int key, std::string displayName) :
        val(val), key(key), displayName(displayName) { cooldown = 400ms; }
};

class ViewportBinding : public InputBinding {
    Viewport& v;
    int upKey, downKey, leftKey, rightKey, zoomInKey, zoomOutKey, resetKey;
    float moveStep, zoomStep;
    std::complex<float> resetPoint;
    float resetZoom;
public:
    virtual bool process(Window& window, float dt, float multiplier) override;
    ViewportBinding(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey,
        int resetKey, float moveStep, float zoomStep) : v(v), upKey(upKey),
        downKey(downKey), leftKey(leftKey), rightKey(rightKey), zoomInKey(zoomInKey), zoomOutKey(zoomOutKey),
        resetKey(resetKey), resetPoint(v.getCenter()), resetZoom(v.getBreadth()), moveStep(moveStep), zoomStep(zoomStep) {}
};

class TriggerBinding : public InputBinding {
    std::function<void()> handler;
    int triggerKey;
public:
    virtual bool process(Window& window, float dt, float multiplier) override;
    TriggerBinding(const std::function<void()>& handler, int triggerKey) : handler(handler), triggerKey(triggerKey) {}
};

class MouseTriggerBinding : public InputBinding {
    std::function<void(double, double)> handler;
    int triggerButton;
public:
    virtual bool process(Window& window, float dt, float multiplier) override;
    MouseTriggerBinding(const std::function<void(double, double)>& handler, int triggerButton) : handler(handler), triggerButton(triggerButton) {}
};

template<typename T>
InputBinding& InputHandler::addScalar(T& val, int upKey, int downKey, T step, const std::string& displayName, T min, T max) {
    auto b = new ScalarBinding(val, upKey, downKey, step, displayName, min, max);
    bindings.push_back(b);
    return *b;
}

template<typename T>
bool ScalarBinding<T>::process(Window& window, float dt, float multiplier) {
    if constexpr(std::is_integral_v<T>)
        dt = 1;

    bool inputReceived = false;
    if(window.isKeyPressed(upKey) && timer.get() > cooldown) {
        val += step * dt * multiplier;
        inputReceived = true;
    }
    if(window.isKeyPressed(downKey) && timer.get() > cooldown) {
        val -= step * dt * multiplier;
        inputReceived = true;
    }

    if(inputReceived) {
        if constexpr(is_ordered_v<T>)
            val = std::clamp(val, min, max);
        std::cout << displayName << ": " << val << std::endl;
        timer.reset();
    }

    return inputReceived;
}