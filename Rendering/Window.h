#pragma once
#include <string>
#include <map>
#include <list>
#include <any>
#include "GLFW/glfw3.h"

class Window {
    int width;
    int height;
    GLFWwindow* handle = nullptr;

    void init(const std::string& title);

public:
    Window(int width, int height, const std::string& title, bool resizable) : width{width}, height{height} {init(title);}
    void setSwapInterval(int interval);
    void swapBuffers();
    void poll();
    bool shouldClose();
    ~Window();

    [[nodiscard]] int getWidth() const;
    [[nodiscard]] int getHeight() const;
    [[nodiscard]] GLFWwindow* getGlfwHandle() const;
    void enableGLDebugMessages(GLDEBUGPROC messageFunc);
    [[nodiscard]] bool isKeyPressed(int key);
};