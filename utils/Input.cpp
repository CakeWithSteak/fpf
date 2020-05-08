#include "Input.h"

bool InputHandler::process(float dt) {
    dt = std::clamp(dt, 0.0f, 1.0f);
    bool inputReceived = false;
    float mult = (multiplierKey.has_value() &&  window.isKeyPressed(*multiplierKey)) ? multiplier : 1.f;

    for(auto binding : bindings) {
        if(binding->process(window, dt, mult))
            inputReceived = true;
    }
    return inputReceived;
}


void InputHandler::addToggle(bool& val, int key, const std::string& displayName) {
    bindings.push_back(new ToggleBinding(val, key, displayName));
}

void InputHandler::addViewport(Viewport& v, int upKey, int downKey, int leftKey, int rightKey, int zoomInKey, int zoomOutKey, int resetKey, float moveStep, float zoomStep) {
    bindings.push_back(new ViewportBinding(v, upKey, downKey, leftKey, rightKey, zoomInKey, zoomOutKey, resetKey, moveStep, zoomStep));
}

void InputHandler::addTrigger(const std::function<void()>& handler, int triggerKey) {
    bindings.push_back(new TriggerBinding(handler, triggerKey));
}

void InputHandler::addTrigger(const std::function<void(double, double)>& handler, int triggerButton) {
    bindings.push_back(new MouseTriggerBinding(handler, triggerButton));
}

InputHandler::~InputHandler() {
    for(auto binding : bindings)
        delete binding;
}

bool ToggleBinding::process(Window& window, float dt, float multiplier) {
    if(window.isKeyPressed(key) && timer.get() > cooldown) {
        val = !val;
        std::cout << displayName << " " << (val ? "on" : "off") << std::endl;
        timer.reset();
        return true;
    }
    return false;
}

bool ViewportBinding::process(Window& window, float dt, float multiplier) {
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

bool TriggerBinding::process(Window& window, float dt, [[maybe_unused]] float multiplier) {
    if(window.isKeyPressed(triggerKey)) {
        handler();
        return true;
    }
    return false;
}

bool MouseTriggerBinding::process(Window& window, float dt, [[maybe_unused]] float multiplier) {
    auto pos = window.tryGetClickPosition(triggerButton);
    if(pos.has_value()) {
        handler(pos->first, pos->second);
        return true;
    }
    return false;
}
