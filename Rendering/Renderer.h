#pragma once

#include <complex>
#include "../Computation/kernel.h"
#include "../utils/Viewport.h"

class Renderer {
    int width;
    int height;
    const Viewport& viewport;
    float tolerance;

    int numBlocks;
    unsigned int texture;
    unsigned int VAO;
    unsigned int shaderProgram;
    int minimumUniform;
    int maximumUniform;
    cudaGraphicsResource_t cudaSurfaceRes = nullptr;
    cudaResourceDesc cudaSurfaceDesc;
    fpdist_t* cudaBuffer = nullptr;

    void init();
    void initTexture();
    void initShaders();
    void initCuda();
    cudaSurfaceObject_t createSurface();
public:
    Renderer(int width, int height, const Viewport& viewport, float tolerance)
            : width(width), height(height), viewport(viewport), tolerance(tolerance) {init();}
    ~Renderer();

    void render(fpdist_t maxIters);
};


