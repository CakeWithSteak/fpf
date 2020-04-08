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
    bool overlayEnabled = false;

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
    cudaGraphicsResource_t cudaBufferRes = nullptr;
    cudaResourceDesc cudaSurfaceDesc;
    dist_t* cudaBuffer = nullptr;
    int* cudaPathLengthPtr = nullptr;
    NvrtcCompiler compiler;
    CUfunction kernel;
    CUfunction pathKernel;

    PerformanceMonitor pm{"CUDA kernel time", "Total render time", "Overlay computation time", "Overlay render time"};
    static constexpr size_t PERF_KERNEL = 0;
    static constexpr size_t PERF_RENDER = 1;
    static constexpr size_t PERF_OVERLAY_GEN = 2;
    static constexpr size_t PERF_OVERLAY_RENDER = 3;
    static constexpr int MAX_PATH_STEPS = 256;

    void init(std::string_view cudaCode);
    void initTexture();
    void initShaders();
    void initCuda(bool registerPathRes = true);
    void initKernels(std::string_view cudaCode);
    cudaSurfaceObject_t createSurface();
public:
    Renderer(int width, int height, const Viewport& viewport, const ModeInfo& mode, std::string_view cudaCode)
            : width(width), height(height), viewport(viewport), mode(mode) {init(cudaCode);}
    ~Renderer();

    void render(dist_t maxIters, float metricArg, const std::complex<float>& p, float colorCutoff);
    std::string getPerformanceReport();
    void resize(int newWidth, int newHeight);
    int generatePathOverlay(const std::complex<float>& z, float tolerance, const std::complex<float>& p);
};


