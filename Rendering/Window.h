#pragma once
#include <string>
#include <map>
#include <list>
#include <functional>
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#if (!(GLFW_VERSION_MAJOR > 3) && (GLFW_VERSION_MINOR < 3))
#warning "GLFW version 3.3 or newer is recommended"
#endif

class Window {
    int width;
    int height;
    GLFWwindow* handle = nullptr;
    std::optional<std::function<void(Window&, int, int)>> resizeCallback;
    void init(const std::string& title, bool resizable, bool visible);
public:
    Window(int width, int height, const std::string& title, bool resizable, bool visible) : width{width}, height{height} {init(title, resizable, visible);}
    void setSwapInterval(int interval);
    void swapBuffers();
    void poll();
    bool shouldClose();
    void setShouldClose(bool val);
    void minimize();
    void restore();

    ~Window();

    [[nodiscard]] int getWidth() const;
    [[nodiscard]] int getHeight() const;
    [[nodiscard]] GLFWwindow* getGlfwHandle() const;
    void enableGLDebugMessages(GLDEBUGPROC messageFunc);
    [[nodiscard]] bool isKeyPressed(int key);
    void setResizeCallback(std::function<void(Window&, int, int)> callback);
    [[nodiscard]] std::optional<std::pair<double, double>> tryGetClickPosition(int button);
    void enforceAspectRatio();
};