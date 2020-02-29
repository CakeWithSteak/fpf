#pragma once

#include <complex>
#include "../Computation/kernel.h"
#include "../utils/Viewport.h"
#include "../Compilation/NvrtcCompiler.h"
#include "../utils/PerformanceMonitor.h"

class Renderer {
    int width;
    int height;
    const Viewport& viewport;

    int numBlocks;
    unsigned int texture;
    unsigned int VAO;
    unsigned int shaderProgram;
    int minimumUniform;
    int maximumUniform;
    cudaGraphicsResource_t cudaSurfaceRes = nullptr;
    cudaResourceDesc cudaSurfaceDesc;
    fpdist_t* cudaBuffer = nullptr;
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
    Renderer(int width, int height, const Viewport& viewport, std::string_view cudaCode)
            : width(width), height(height), viewport(viewport) {init(cudaCode);}
    ~Renderer();

    void render(fpdist_t maxIters, float tolerance);
    std::string getPerformanceReport();
};


