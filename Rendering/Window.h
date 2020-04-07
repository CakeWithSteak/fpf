#pragma once
#include <string>
#include <map>
#include <list>
#include <functional>
#include "GLFW/glfw3.h"

class Window {
    int width;
    int height;
    GLFWwindow* handle = nullptr;
    std::optional<std::function<void(Window&, int, int)>> resizeCallback;
    void init(const std::string& title, bool resizable);

public:
    Window(int width, int height, const std::string& title, bool resizable) : width{width}, height{height} {init(title, resizable);}
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
};