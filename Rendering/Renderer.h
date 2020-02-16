#pragma once

#include <complex>
#include "../Computation/kernel.h"
#include "../utils/Viewport.h"
#include "../Compilation/NvrtcCompiler.h"

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

    void init();
    void initTexture();
    void initShaders();
    void initCuda();
    void initKernel();
    cudaSurfaceObject_t createSurface();
public:
    Renderer(int width, int height, const Viewport& viewport)
            : width(width), height(height), viewport(viewport) {init();}
    ~Renderer();

    void render(fpdist_t maxIters, float tolerance);
};


