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
#include "cli.h"
#include "controls.h"


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

std::string getCudaCode(const std::string& expr) {
    auto cudaCode = compileExpression(expr);
    std::cout << "CUDA expression: " << cudaCode << "\n\n" << std::flush;
    auto finalCode = runtimeTemplateCode + cudaCode + "}";
    return finalCode;
}

//Precision very low at (1.0067,-1.219) -> (1.00677,-1.21893)

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    Options opt = getOptions(argc, argv);
    const ModeInfo mode = opt.mode;

    int maxIters = 128;
    float metricArg = mode.argInitValue;
    std::complex<float> p{0};
    bool colorCutoffEnabled = (mode.defaultColorCutoff != -1);
    float colorCutoff = colorCutoffEnabled ? mode.defaultColorCutoff : 10.0f;

    auto cudaCode = getCudaCode(opt.expression);

    Viewport viewport(0, 2);

    Window window(opt.width, opt.height, "Fixed point fractals - " + mode.displayName, false);
    window.setSwapInterval(1);
    window.enableGLDebugMessages(glDebugCallback);

    InputHandler input = initControls(window, viewport, mode, maxIters, metricArg, p, colorCutoffEnabled, colorCutoff);

    Renderer renderer(opt.width, opt.height, viewport, mode, cudaCode);

    //First render
    glClear(GL_COLOR_BUFFER_BIT);
    renderer.render(maxIters, metricArg, p, colorCutoffEnabled ? colorCutoff : std::numeric_limits<float>::max());
    window.swapBuffers();
    window.poll();

    Timer timer;
    while(!window.shouldClose()) {
        bool repaintNeeded = input.process(timer.getSeconds());
        timer.reset();

        if(repaintNeeded) {
            glClear(GL_COLOR_BUFFER_BIT);
            float actualColorCutoff = colorCutoffEnabled ? colorCutoff : std::numeric_limits<float>::max();
            renderer.render(maxIters, metricArg, p, actualColorCutoff);
            window.poll(); // The Renderer call may take a long time, so we poll here to ensure responsiveness
            window.swapBuffers();
        }
        window.poll();
        if(!repaintNeeded)
            std::this_thread::sleep_for(40ms);
    }
    std::cout << "\n" << renderer.getPerformanceReport();
}
