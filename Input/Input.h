#pragma once
#include "../Rendering/Window.h"
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <concepts>
#include "../utils/Viewport.h"
#include "../utils/Timer.h"
#include "Controller.h"

using namespace std::chrono_literals;

template <typename T>
using limits = std::numeric_limits<T>;


//Types for callbacks from TriggerBindings. The return value determines whether or not to redraw the screen.
template <class T>
concept TriggerAction = requires(T t) {{t()} -> std::same_as<void>;} || requires(T t) {{t()} -> std::same_as<bool>;};

template <class T>
concept MouseTriggerAction = requires(T t) {{t(1.0, 1.0)} -> std::same_as<void>;} || requires(T t) {{t(1.0, 1.0)} -> std::same_as<bool>;};

class InputBinding {
protected:
    Timer timer;
    std::chrono::milliseconds cooldown = 0ms;
    bool isEnabled = true;
public:
    virtual bool process(Window& window, double dt, double multiplier) = 0;
    virtual ~InputBinding() = default;
    virtual void setCooldown(const std::chrono::milliseconds& val) {
        cooldown = val;
    }
    virtual void setEnabled(bool enabled) {
        isEnabled = enabled;
    }
};

class InputHandler : public Controller {
    Window& window;
    std::vector<InputBinding*> bindings;
    std::optional<int> multiplierKey;
    double multiplier = 1;
public:
    bool process(double dt) override;

    template <typename T>
    InputBinding& addScalar(T& val, int upKey, int downKey, T step, const std::string& displayName, T min = limits<T>::lowest(), T max = limits<T>::max());

    InputBinding& addToggle(bool& val, int key, const std::string& displayName);
    InputBinding& addViewport(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey, int resetKey, double moveStep, double zoomStep);
    template <TriggerAction T>
    InputBinding& addTrigger(const T& handler, int triggerKey);
    template <MouseTriggerAction T>
    InputBinding& addTrigger(const T& handler, int triggerButton);

    InputHandler(Window& window, const std::optional<int>& multiplierKey = {}, double multiplier = 1) : window(window),
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
    virtual bool process(Window& window, double dt, double multiplier) override;
    ScalarBinding(T& val, int upKey, int downKey, T step, std::string displayName, T min = limits<T>::lowest(), T max = limits<T>::max()) :
        val(val), min(min), max(max), upKey(upKey), downKey(downKey), step(step), displayName(displayName) {}
};

class ToggleBinding : public InputBinding {
    bool& val;
    int key;
    std::string displayName;
public:
    virtual bool process(Window& window, double dt, double multiplier) override;
    ToggleBinding(bool& val, int key, std::string displayName) :
        val(val), key(key), displayName(displayName) { cooldown = 400ms; }
};

class ViewportBinding : public InputBinding {
    Viewport& v;
    int upKey, downKey, leftKey, rightKey, zoomInKey, zoomOutKey, resetKey;
    double moveStep, zoomStep;
    std::complex<double> resetPoint;
    double resetZoom;
public:
    virtual bool process(Window& window, double dt, double multiplier) override;
    ViewportBinding(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey,
        int resetKey, double moveStep, double zoomStep) : v(v), upKey(upKey),
        downKey(downKey), leftKey(leftKey), rightKey(rightKey), zoomInKey(zoomInKey), zoomOutKey(zoomOutKey),
        resetKey(resetKey), resetPoint(v.getCenter()), resetZoom(v.getBreadth()), moveStep(moveStep), zoomStep(zoomStep) {}
};

template <TriggerAction T>
class TriggerBinding : public InputBinding {
    T handler;
    int triggerKey;

    using HandlerReturnType = decltype(std::declval<T>()());
public:
    virtual bool process(Window& window, double dt, double multiplier) override;
    TriggerBinding(const T& handler, int triggerKey) : handler(handler), triggerKey(triggerKey) {}

};

template <MouseTriggerAction T>
class MouseTriggerBinding : public InputBinding {
    T handler;
    int triggerButton;

    using HandlerReturnType = decltype(std::declval<T>()(1.0, 1.0));
public:
    virtual bool process(Window& window, double dt, double multiplier) override;
    MouseTriggerBinding(const T& handler, int triggerButton) : handler(handler), triggerButton(triggerButton) {}
};

template<typename T>
InputBinding& InputHandler::addScalar(T& val, int upKey, int downKey, T step, const std::string& displayName, T min, T max) {
    auto b = new ScalarBinding(val, upKey, downKey, step, displayName, min, max);
    bindings.push_back(b);
    return *b;
}

template <TriggerAction T>
InputBinding& InputHandler::addTrigger(const T& handler, int triggerKey) {
    auto b = new TriggerBinding(handler, triggerKey);
    bindings.push_back(b);
    return *b;
}

template <MouseTriggerAction T>
InputBinding& InputHandler::addTrigger(const T& handler, int triggerKey) {
    auto b = new MouseTriggerBinding(handler, triggerKey);
    bindings.push_back(b);
    return *b;
}

template<typename T>
bool ScalarBinding<T>::process(Window& window, double dt, double multiplier) {
    if(!isEnabled)
        return false;

    if constexpr(std::integral<T>)
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
        if constexpr(std::totally_ordered<T>)
            val = std::clamp(val, min, max);
        std::cout << displayName << ": " << val << std::endl;
        timer.reset();
    }

    return inputReceived;
}

template <TriggerAction T>
bool TriggerBinding<T>::process(Window& window, double dt, [[maybe_unused]] double multiplier) {
    if(!isEnabled)
        return false;
    if(window.isKeyPressed(triggerKey) && timer.get() > cooldown) {
        if constexpr (std::same_as<HandlerReturnType, void>) {
            handler();
            timer.reset();
            return true;
        } else {
            auto handlerVal = handler();
            timer.reset();
            return handlerVal;
        }
    }
    return false;
}

template <MouseTriggerAction T>
bool MouseTriggerBinding<T>::process(Window& window, double dt, [[maybe_unused]] double multiplier) {
    if(!isEnabled)
        return false;
    auto pos = window.tryGetClickPosition(triggerButton);
    if(pos.has_value() && timer.get() > cooldown) {
        if constexpr (std::same_as<HandlerReturnType, void>) {
            handler(pos->first, pos->second);
            timer.reset();
            return true;
        } else {
            auto handlerVal = handler(pos->first, pos->second);
            timer.reset();
            return handlerVal;
        }
    }
    return false;
}