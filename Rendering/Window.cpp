#include <iostream>
#include <algorithm>
#include <cmath>
#include "Window.h"

void Window::init(const std::string& title, bool resizable, bool visible) {
    glfwSetErrorCallback([]([[maybe_unused]] int unused, const char* err) {
        std::cerr << "GLFW error: " << err << std::endl;
    });

    glfwInit();
    std::atexit(glfwTerminate);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, resizable);
    glfwWindowHint(GLFW_VISIBLE, visible);

    handle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if(!handle) {
        throw std::runtime_error("Failed to open window");
    }

    glfwMakeContextCurrent(handle);

    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
    glViewport(0, 0, width, height);
    glClearColor(0, 0, 0, 0);

    glfwSetWindowUserPointer(handle, this);

    if(resizable) {
        glfwSetWindowSizeCallback(handle, [](GLFWwindow* handle, int newWidth, int newHeight){
            if(newWidth == 0 || newHeight == 0)
                return; //On Windows the window is resized to 0x0 when minimising, which would mess up Renderer state
            glViewport(0, 0, newWidth, newHeight);
            Window& window = *static_cast<Window*>(glfwGetWindowUserPointer(handle));
            if(window.width * newHeight != window.height * newWidth) {
                newHeight = std::min(newWidth, newHeight);
                glfwSetWindowSize(handle, std::round((window.width / (float)(window.height)) * newHeight), newHeight);
                return;
            }
            window.width = newWidth;
            window.height = newHeight;
            if(window.resizeCallback.has_value())
                window.resizeCallback.value()(window, newWidth, newHeight);
        });

        glfwSetWindowAspectRatio(handle, width, height);
        glfwSetWindowSizeLimits(handle, 100, 100, GLFW_DONT_CARE, GLFW_DONT_CARE);
    }

    glfwSetWindowPosCallback(handle, [](GLFWwindow* handle, int x, int y) {
        Window& window = *static_cast<Window*>(glfwGetWindowUserPointer(handle));
        if(window.moveCallback.has_value())
            window.moveCallback.value()(window, x, y);
    });

    glfwSetWindowMaximizeCallback(handle, [](GLFWwindow* handle, int maximized) {
        Window& window = *static_cast<Window*>(glfwGetWindowUserPointer(handle));
        if(window.maximizeCallback.has_value())
            window.maximizeCallback.value()(window, maximized);
    });
}

void Window::setSwapInterval(int interval) {
    glfwSwapInterval(interval);
}

void Window::swapBuffers() {
    glfwSwapBuffers(handle);
}

void Window::poll() {
    glfwPollEvents();
}

Window::~Window() {
    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(handle);
}

int Window::getWidth() const {
    return width;
}

int Window::getHeight() const {
    return height;
}

GLFWwindow* Window::getGlfwHandle() const {
    return handle;
}

bool Window::shouldClose() {
    return glfwWindowShouldClose(handle);
}

void Window::enableGLDebugMessages(GLDEBUGPROC messageFunc) {
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(messageFunc, nullptr);
#ifdef NDEBUG
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
#else
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif
}

bool Window::isKeyPressed(int key) {
    return glfwGetKey(handle, key) == GLFW_PRESS;
}

void Window::setShouldClose(bool val) {
    glfwSetWindowShouldClose(handle, val);
}

void Window::minimize() {
    glfwIconifyWindow(handle);
}

void Window::restore() {
    glfwRestoreWindow(handle);
    glfwFocusWindow(handle);
}

void Window::setResizeCallback(std::function<void(Window&, int, int)> callback) {
    resizeCallback = callback;
}

std::optional<std::pair<double, double>> Window::tryGetClickPosition(int button) {
    if(glfwGetMouseButton(handle, button) == GLFW_PRESS) {
        double x,y;
        glfwGetCursorPos(handle, &x, &y);
        return {{x, y}};
    }
    return {};
}

//Resizes the window to the closest dimension possible while maintaining the aspect ratio and staying on screen.
void Window::clipToScreen() {
    if(!resizeable)
        return;

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    int waWidth, waHeight;
    glfwGetMonitorWorkarea(monitor, nullptr, nullptr, &waWidth, &waHeight);
    int leftBorder, rightBorder, topBorder, bottomBorder;
    glfwGetWindowFrameSize(handle, &leftBorder, &topBorder, &rightBorder, &bottomBorder);

    int maxWidth = waWidth - (leftBorder + rightBorder);
    int maxHeight = waHeight - (topBorder + bottomBorder);

    if(height <= maxHeight && width <= maxWidth)
        return;

    int newHeight = std::min(std::min(maxWidth, maxHeight), width);
    glfwSetWindowSize(handle, (width / static_cast<float>(height)) * newHeight, newHeight);
    poll();
}

void Window::setMoveCallback(std::function<void(Window &, int, int)> callback) {
    moveCallback = callback;
}

void Window::setMaximizeCallback(std::function<void(Window &, bool)> callback) {
    maximizeCallback = callback;
}
