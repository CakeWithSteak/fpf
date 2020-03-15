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
    dist_t* cudaBuffer = nullptr;
    NvrtcCompiler compiler;
    CUfunction kernel;

    PerformanceMonitor pm{"CUDA kernel time", "Total render time"};
    static constexpr size_t PERF_KERNEL = 0;
    static constexpr size_t PERF_RENDER = 1;

    void init(std::string_view cudaCode, DistanceMetric metric);
    void initTexture();
    void initShaders();
    void initCuda();
    void initKernel(std::string_view cudaCode, DistanceMetric metric);
    cudaSurfaceObject_t createSurface();
public:
    Renderer(int width, int height, const Viewport& viewport, DistanceMetric metric, std::string_view cudaCode)
            : width(width), height(height), viewport(viewport) {init(cudaCode, metric);}
    ~Renderer();

    void render(dist_t maxIters, float tolerance, const std::complex<float>& p);
    std::string getPerformanceReport();
};


