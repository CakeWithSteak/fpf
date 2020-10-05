#include "Input.h"

bool InputHandler::process(double dt) {
    dt = std::clamp(dt, 0.0, 1.0);
    bool inputReceived = false;
    double mult = (multiplierKey.has_value() &&  window.isKeyPressed(*multiplierKey)) ? multiplier : 1;

    for(auto binding : bindings) {
        if(binding->process(window, dt, mult))
            inputReceived = true;
    }
    return inputReceived;
}


InputBinding&  InputHandler::addToggle(bool& val, int key, const std::string& displayName) {
    auto b = new ToggleBinding(val, key, displayName);
    bindings.push_back(b);
    return *b;
}

InputBinding&  InputHandler::addViewport(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey, int resetKey, double moveStep, double zoomStep) {
    auto b = new ViewportBinding(v, upKey, downKey, leftKey, rightKey, zoomInKey, zoomOutKey, resetKey, moveStep, zoomStep);
    bindings.push_back(b);
    return *b;
}

InputBinding&  InputHandler::addTrigger(const std::function<void()>& handler, int triggerKey) {
    auto b = new TriggerBinding(handler, triggerKey);
    bindings.push_back(b);
    return *b;
}

InputBinding&  InputHandler::addTrigger(const std::function<void(double, double)>& handler, int triggerButton) {
    auto b = new MouseTriggerBinding(handler, triggerButton);
    bindings.push_back(b);
    return *b;
}

InputHandler::~InputHandler() {
    for(auto binding : bindings)
        delete binding;
}

bool ToggleBinding::process(Window& window, double dt, double multiplier) {
    if(window.isKeyPressed(key) && timer.get() > cooldown) {
        val = !val;
        std::cout << displayName << " " << (val ? "on" : "off") << std::endl;
        timer.reset();
        return true;
    }
    return false;
}

bool ViewportBinding::process(Window& window, double dt, double multiplier) {
    if(!isEnabled)
        return false;
    bool inputReceived = false;
    if(window.isKeyPressed(upKey)) {
        v.move(Viewport::Direction::UP, moveStep * dt * multiplier);
        inputReceived = true;
    }
    if(window.isKeyPressed(downKey)) {
        v.move(Viewport::Direction::DOWN, moveStep * dt * multiplier);
        inputReceived = true;
    }
    if(window.isKeyPressed(leftKey)) {
        v.move(Viewport::Direction::LEFT, moveStep * dt * multiplier);
        inputReceived = true;
    }
    if(window.isKeyPressed(rightKey)) {
        v.move(Viewport::Direction::RIGHT, moveStep * dt * multiplier);
        inputReceived = true;
    }
    if(window.isKeyPressed(zoomInKey)) {
        v.zoom(zoomStep * dt * multiplier);
        inputReceived = true;
    }
    if(window.isKeyPressed(zoomOutKey)) {
        v.zoom(-zoomStep * dt * multiplier);
        inputReceived = true;
    }
    if(window.isKeyPressed(resetKey)) {
        v.zoomTo(resetZoom);
        v.moveTo(resetPoint);
        inputReceived = true;
    }
    return inputReceived;
}

bool TriggerBinding::process(Window& window, double dt, [[maybe_unused]] double multiplier) {
    if(!isEnabled)
        return false;
    if(window.isKeyPressed(triggerKey) && timer.get() > cooldown) {
        handler();
        timer.reset();
        return true;
    }
    return false;
}

bool MouseTriggerBinding::process(Window& window, double dt, [[maybe_unused]] double multiplier) {
    if(!isEnabled)
        return false;
    auto pos = window.tryGetClickPosition(triggerButton);
    if(pos.has_value() && timer.get() > cooldown) {
        handler(pos->first, pos->second);
        timer.reset();
        return true;
    }
    return false;
}
