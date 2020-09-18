#include <iostream>
#include <thread>
#include "Rendering/Window.h"
#include "Rendering/Renderer.h"
#include "utils/Viewport.h"
#include "utils/Timer.h"
#include "Compilation/compileExpression.h"
#include "Computation/runtime_template.h"
#include "cli.h"
#include "controls.h"
#include "utils/serialization.h"
#include "Input/Animator.h"


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
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204 || id == 131186) return;

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

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    Options opt = getOptions(argc, argv);
    State state;

    if(opt.deserializationPath.has_value()) {
        state = deserialize(*opt.deserializationPath);
    } else {
        state = State(opt);
    }
    bool animating = opt.animParams.has_value();

    auto cudaCode = getCudaCode(state.expr);

    Window window(state.width, state.height, "Fixed point fractals - " + state.mode.displayName, !animating);
    window.setSwapInterval(1);
    window.enableGLDebugMessages(glDebugCallback);

    Renderer renderer(state.width, state.height, state.viewport, state.mode, cudaCode, opt.doublePrec);

    RuntimeState runtimeState{.window = window, .renderer = renderer, .refsPath = opt.refsPath};

    window.setResizeCallback([&state, &runtimeState](Window& win, int width, int height){
        state.width = width;
        state.height = height;
        runtimeState.renderer.resize(width, height);
        runtimeState.forceRerender = true;
    });

    std::unique_ptr<Controller> control;
    if(animating)
        control = std::make_unique<Animator>(*opt.animParams, state, runtimeState);
    else
        control = initControls(state, runtimeState);

    if(state.pathStart.has_value())
        renderer.generatePath(state.pathStart.value(), state.metricArg, state.p);

    if(state.lineTransEnabled)
        renderer.generateLineTransform(*state.lineTransStart, *state.lineTransEnd, state.lineTransIteration, state.p);

    //First render
    runtimeState.forceRerender = true;

    Timer timer;
    while(!window.shouldClose()) {
        window.poll();
        bool repaintNeeded = control->process(timer.getSeconds()) || runtimeState.forceRerender;
        timer.reset();

        if(repaintNeeded) {
            glClear(GL_COLOR_BUFFER_BIT);
            float actualColorCutoff = state.colorCutoffEnabled ? state.colorCutoff : std::numeric_limits<float>::max();
            renderer.render(state.maxIters, state.metricArg, state.p, actualColorCutoff);
            window.swapBuffers();
            runtimeState.forceRerender = false;
        }
        else
            std::this_thread::sleep_for(40ms);
    }
    std::cout << "\n" << renderer.getPerformanceReport();
}
