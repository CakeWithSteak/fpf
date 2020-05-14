#pragma once
#include <complex>
#include "../Computation/kernel.h"
#include "../utils/Viewport.h"
#include "../Compilation/NvrtcCompiler.h"
#include "../utils/PerformanceMonitor.h"
#include "../modes.h"
#include "Shader.h"
#include "shaders.h"

class Renderer {
    int width;
    int height;
    const Viewport& viewport;
    ModeInfo mode;

    bool pathEnabled = false;
    std::complex<float> pathStart;
    std::complex<float> lastP;
    float lastTolerance;

    bool lineTransEnabled = false;
    std::complex<float> lineTransStart;
    std::complex<float> lineTransEnd;
    int lineTransIteration;

    int numBlocks;
    unsigned int texture;
    unsigned int mainVAO;
    unsigned int overlayVAO;
    unsigned int overlayLineVBO;

    Shader mainShader{mainVertexShaderCode, mainFragmentShaderCode};
    int minimumUniform;
    int maximumUniform;
    Shader overlayShader{overlayVertexShaderCode, overlayFragmentShaderCode};
    int viewCenterUniform;
    int viewBreadthUniform;

    cudaGraphicsResource_t cudaSurfaceRes = nullptr;
    cudaGraphicsResource_t overlayBufferRes = nullptr;
    cudaResourceDesc cudaSurfaceDesc;
    dist_t* cudaBuffer = nullptr;
    int* cudaPathLengthPtr = nullptr;
    NvrtcCompiler compiler;
    CUfunction kernel;
    CUfunction pathKernel;
    CUfunction lineTransformKernel;

    PerformanceMonitor pm{"Main kernel time", "Total render time", "Orbit compute time", "Overlay render time", "Line transform compute time"};
    static constexpr size_t PERF_KERNEL = 0;
    static constexpr size_t PERF_RENDER = 1;
    static constexpr size_t PERF_OVERLAY_GEN = 2;
    static constexpr size_t PERF_OVERLAY_RENDER = 3;
    static constexpr size_t PERF_LINE_TRANS_GEN = 4;

    static constexpr int MAX_PATH_STEPS = 256;
    static constexpr float PATH_PARAM_UPDATE_THRESHOLD = 0.01f;
    static constexpr float PATH_TOL_UPDATE_THRESHOLD = 0.001f;
    static constexpr float DEFAULT_PATH_TOLERANCE = 0.001f;

    static constexpr int LINE_TRANS_NUM_POINTS = 100'000;

    void init(std::string_view cudaCode);
    void initTexture();
    void initShaders();
    void initCuda(bool registerPathRes = true);
    void initKernels(std::string_view cudaCode);
    cudaSurfaceObject_t createSurface();
    void refreshOverlayIfNeeded(const std::complex<float>& p, float metricArg);
    void generateLineTransformImpl(const std::complex<float>& p);
    inline bool isOverlayEnabled() { return pathEnabled || lineTransEnabled; }
    int getOverlayLength();
public:
    Renderer(int width, int height, const Viewport& viewport, const ModeInfo& mode, std::string_view cudaCode)
            : width(width), height(height), viewport(viewport), mode(mode) {init(cudaCode);}
    ~Renderer();

    void render(dist_t maxIters, float metricArg, const std::complex<float>& p, float colorCutoff);
    std::string getPerformanceReport();
    void resize(int newWidth, int newHeight);
    int generatePath(const std::complex<float>& z, float tolerance, const std::complex<float>& p);
    void hideOverlay();
    std::vector<unsigned char> exportImageData();
    void generateLineTransform(const std::complex<float>& start, const std::complex<float>& end, int iteration,
                               const std::complex<float>& p);
    void setLineTransformIteration(int iteration, const std::complex<float>& p);
};


