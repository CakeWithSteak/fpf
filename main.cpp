#include <iostream>
#include <algorithm>
#include <thread>
#include "Rendering/Window.h"
#include "Rendering/Renderer.h"
#include "utils/Viewport.h"
#include "utils/Timer.h"
#include "Compilation/compileExpression.h"
#include "Computation/runtime_template.h"
#include "utils/ModeInfo.h"
#include "modes.h"


using namespace std::chrono_literals;

//https://learnopengl.com/In-Practice/Debugging
void APIENTRY glDebugCallback(GLenum source,
                            GLenum type,
                            GLuint id,
                            GLenum severity,
                            GLsizei length,
                            const GLchar *message,
                            const void *userParam)
{
    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " <<  message << std::endl;

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
    } std::cout << std::endl;

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
    } std::cout << std::endl;

    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
    } std::cout << std::endl;
    std::cout << std::endl;
}

bool handleInputs(Window& window, Viewport& viewport, int& maxIters, float& metricArg, std::complex<float>& p, float deltaTime, float moveStep, float zoomStep, float pStep, const ModeInfo& mode) {
    deltaTime = std::clamp(deltaTime, 0.0f, 1.0f);
    bool inputReceived = false;

    if(window.isKeyPressed(GLFW_KEY_UP)) {
        viewport.move(Viewport::Direction::UP, moveStep * deltaTime);
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_DOWN)) {
        viewport.move(Viewport::Direction::DOWN, moveStep * deltaTime);
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_LEFT)) {
        viewport.move(Viewport::Direction::LEFT, moveStep * deltaTime);
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_RIGHT)) {
        viewport.move(Viewport::Direction::RIGHT, moveStep * deltaTime);
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_KP_ADD) || window.isKeyPressed(GLFW_KEY_LEFT_SHIFT)) {
        viewport.zoom(zoomStep * deltaTime);
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_KP_SUBTRACT) || window.isKeyPressed(GLFW_KEY_LEFT_CONTROL)) {
        viewport.zoom(-zoomStep * deltaTime);
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_HOME)) {
        viewport.zoomTo(2);
        viewport.moveTo({0,0});
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_KP_MULTIPLY) || window.isKeyPressed(GLFW_KEY_RIGHT_BRACKET)) {
        maxIters += 2;
        maxIters = std::clamp(maxIters, 1, 1024);
        std::cout << "Max iterations: " << maxIters << std::endl;
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_KP_DIVIDE) || window.isKeyPressed(GLFW_KEY_LEFT_BRACKET)) {
        maxIters -= 2;
        maxIters = std::clamp(maxIters, 1, 1024);
        std::cout << "Max iterations: " << maxIters << std::endl;
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_0)) {
        metricArg += mode.argStep;
        metricArg = std::clamp(metricArg, mode.argMin, mode.argMax);
        std::cout << mode.argDisplayName << ": " << metricArg << std::endl;
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_9)) {
        metricArg -= mode.argStep;
        metricArg = std::clamp(metricArg, mode.argMin, mode.argMax);
        std::cout << mode.argDisplayName << ": " << metricArg << std::endl;
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_W)) {
        p.imag(p.imag() + pStep * deltaTime);
        std::cout << "p = " << p << std::endl;
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_S)) {
        p.imag(p.imag() - pStep * deltaTime);
        std::cout << "p = " << p << std::endl;
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_D)) {
        p.real(p.real() + pStep * deltaTime);
        std::cout << "p = " << p << std::endl;
        inputReceived = true;
    }
    if(window.isKeyPressed(GLFW_KEY_A)) {
        p.real(p.real() - pStep * deltaTime);
        std::cout << "p = " << p << std::endl;
        inputReceived = true;
    }

    return inputReceived;
}

std::string getCudaCode() {
    std::cout << "Expression> ";
    std::string expr;
    std::getline(std::cin, expr);
    auto cudaCode = compileExpression(expr);
    std::cout << "CUDA expression: " << cudaCode << "\n\n" << std::flush;
    auto finalCode = runtimeTemplateCode + cudaCode + "}";
    return finalCode;
}

ModeInfo getMode() {
    DistanceMetric metric = JULIA; //todo cli
    auto mode = modes.at(metric);
    return mode;
}

//Precision very low at (1.0067,-1.219) -> (1.00677,-1.21893)

int main() {
    std::ios::sync_with_stdio(false);
    constexpr int WIN_HEIGHT = 1024;
    constexpr int WIN_WIDTH = 1024;
    constexpr float MOVE_STEP = 2.0f;
    constexpr float ZOOM_STEP = 0.4f;
    constexpr float PARAM_STEP = 0.05f;

    ModeInfo mode = getMode();

    int maxIters = 128;
    float metricArg = mode.argInitValue;
    std::complex<float> p{0};

    auto cudaCode = getCudaCode();

    Viewport viewport(0, 2);

    Window window(WIN_WIDTH, WIN_HEIGHT, "Fixed point fractals - " + mode.displayName, false);
    window.setSwapInterval(1);
    window.enableGLDebugMessages(glDebugCallback);

    Renderer renderer(WIN_WIDTH, WIN_HEIGHT, viewport, mode, cudaCode);

    //First render
    glClear(GL_COLOR_BUFFER_BIT);
    renderer.render(maxIters, metricArg, p);
    window.swapBuffers();
    window.poll();

    Timer timer;
    while(!window.shouldClose()) {
        bool repaintNeeded = handleInputs(window, viewport, maxIters, metricArg, p, timer.getSeconds(), MOVE_STEP, ZOOM_STEP, PARAM_STEP, mode);
        timer.reset();

        if(repaintNeeded) {
            glClear(GL_COLOR_BUFFER_BIT);
            renderer.render(maxIters, metricArg, p);
            window.poll(); // The Renderer call may take a long time, so we poll here to ensure responsiveness
            window.swapBuffers();
        }
        window.poll();
        if(!repaintNeeded)
            std::this_thread::sleep_for(40ms);
    }
    std::cout << "\n" << renderer.getPerformanceReport();
}
