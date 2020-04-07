#pragma once
#include <complex>
#include "../Computation/kernel.h"
#include "../utils/Viewport.h"
#include "../Compilation/NvrtcCompiler.h"
#include "../utils/PerformanceMonitor.h"
#include "../modes.h"

class Renderer {
    int width;
    int height;
    const Viewport& viewport;
    ModeInfo mode;

    int numBlocks;
    unsigned int texture;
    unsigned int VAO;
    unsigned int shaderProgram;
    int minimumUniform;
    int maximumUniform;
    cudaGraphicsResource_t cudaSurfaceRes = nullptr;
    cudaResourceDesc cudaSurfaceDesc;
    dist_t* cudaBuffer = nullptr;
    NvrtcCompiler compiler;
    CUfunction kernel;

    PerformanceMonitor pm{"CUDA kernel time", "Total render time"};
    static constexpr size_t PERF_KERNEL = 0;
    static constexpr size_t PERF_RENDER = 1;

    void init(std::string_view cudaCode);
    void initTexture();
    void initShaders();
    void initCuda();
    void initKernel(std::string_view cudaCode);
    cudaSurfaceObject_t createSurface();
public:
    Renderer(int width, int height, const Viewport& viewport, const ModeInfo& mode, std::string_view cudaCode)
            : width(width), height(height), viewport(viewport), mode(mode) {init(cudaCode);}
    ~Renderer();

    void render(dist_t maxIters, float metricArg, const std::complex<float>& p, float colorCutoff);
    std::string getPerformanceReport();
    void resize(int newWidth, int newHeight);
};


