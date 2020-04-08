#include "glad/glad.h"
#include "Renderer.h"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <driver_types.h>
#include <cassert>
#include "../Computation/safeCall.h"

std::pair<dist_t, dist_t> interleavedMinmax(const dist_t* buffer, size_t size);

float data[] = {
    //XY position and UV coordinates

    -1,  1,  0, 0,  //top left
    -1, -1,  0, 1, //bottom left
     1,  1,  1, 0, //top right

     1,  1,  1, 0, //top right
    -1, -1,  0, 1, //bottom left
     1, -1,  1, 1, //bottom right
};

void Renderer::init(std::string_view cudaCode) {
    unsigned int VAOs[2];
    glGenVertexArrays(2, VAOs);
    mainVAO = VAOs[0];
    overlayVAO = VAOs[1];

    unsigned int VBOs[2];
    glGenBuffers(2, VBOs);
    unsigned int mainVBO = VBOs[0];
    overlayLineVBO = VBOs[1];

    //Init overlay structures
    glBindVertexArray(overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, overlayLineVBO);

    //glBufferStorage(GL_ARRAY_BUFFER, 2 * MAX_PATH_STEPS * sizeof(float), nullptr, GL_MAP_WRITE_BIT); todo
    glBufferData(GL_ARRAY_BUFFER, 2 * MAX_PATH_STEPS * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    //Line vertices
    glVertexAttribPointer(0, 2, GL_FLOAT, false, sizeof(float) * 2, nullptr);
    glEnableVertexAttribArray(0);

    //Init main structures
    glBindVertexArray(mainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mainVBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    //Position
    glVertexAttribPointer(0, 2, GL_FLOAT, false, sizeof(float) * 4, nullptr);
    glEnableVertexAttribArray(0);

    //UV
    glVertexAttribPointer(1, 2, GL_FLOAT, false, sizeof(float) * 4, (void*)(sizeof(float) * 2));
    glEnableVertexAttribArray(1);

    glLineWidth(2);
    glPointSize(4);

    initTexture();
    initShaders();
    initCuda();
    initKernels(cudaCode);
}

void Renderer::initTexture() {
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    //Allocates one-channel float32 texture
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void Renderer::initShaders() {
    mainShader.use();

    mainShader.setUniform("distances", 0);
    mainShader.setUniform("maxHue", mode.maxHue);

    minimumUniform = mainShader.getUniformLocation("minDist");
    maximumUniform = mainShader.getUniformLocation("maxDist");
    viewCenterUniform = overlayShader.getUniformLocation("viewportCenter");
    viewBreadthUniform = overlayShader.getUniformLocation("viewportBreadth");

    if(mode.staticMinMax.has_value()) {
        mainShader.setUniform(minimumUniform, mode.staticMinMax->first);
        mainShader.setUniform(maximumUniform, mode.staticMinMax->second);
    }
}

void Renderer::initCuda(bool registerPathRes) {
    CUDA_SAFE(cudaSetDevice(0));
    numBlocks = (width * height + 1024 - 1) / 1024; //Ceiled division
    CUDA_SAFE(cudaMallocManaged(&cudaBuffer, 2 * numBlocks * sizeof(int))); //Buffer for min/max fpdist values
    CUDA_SAFE(cudaGraphicsGLRegisterImage(&cudaSurfaceRes, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    cudaSurfaceDesc.resType = cudaResourceTypeArray;

    if(registerPathRes) {
        CUDA_SAFE(cudaMallocManaged(&cudaPathLengthPtr, sizeof(int)));
        CUDA_SAFE(cudaGraphicsGLRegisterBuffer(&cudaBufferRes, overlayLineVBO, cudaGraphicsRegisterFlagsWriteDiscard));
    }
}


Renderer::~Renderer() {
    CUDA_SAFE(cudaGraphicsUnregisterResource(cudaSurfaceRes));
    CUDA_SAFE(cudaFree(cudaBuffer));
}

void Renderer::render(dist_t maxIters, float metricArg, const std::complex<float>& p, float colorCutoff) {
    pm.enter(PERF_RENDER);
    auto [start, end] = viewport.getCorners();

    CUDA_SAFE(cudaGraphicsMapResources(1, &cudaSurfaceRes));
    auto surface = createSurface();

    pm.enter(PERF_KERNEL);
    launch_kernel(kernel, start.real(), end.real(), start.imag(), end.imag(), maxIters, cudaBuffer, surface, width, height, p.real(), p.imag(), prepMetricArg(mode.metric, metricArg));
    CUDA_SAFE(cudaDeviceSynchronize());
    pm.exit(PERF_KERNEL);

    CUDA_SAFE(cudaDestroySurfaceObject(surface));
    CUDA_SAFE(cudaGraphicsUnmapResources(1, &cudaSurfaceRes));

    mainShader.use();
    if(!mode.staticMinMax.has_value()) {
        auto [min, max] = interleavedMinmax(cudaBuffer, 2 * numBlocks);
        mainShader.setUniform(minimumUniform, min);
        mainShader.setUniform(maximumUniform, std::min(max, colorCutoff));
        std::cout << "Min: " << min << " max: " << max << "\n";
    }

    glBindVertexArray(mainVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    if(pathEnabled) {
        refreshPathIfNeeded(p, metricArg);
        pm.enter(PERF_OVERLAY_RENDER);
        glBindVertexArray(overlayVAO);
        overlayShader.use();
        overlayShader.setUniform(viewCenterUniform, viewport.getCenter().real(), viewport.getCenter().imag());
        overlayShader.setUniform(viewBreadthUniform, viewport.getBreadth());
        glDrawArrays(GL_LINE_STRIP, 0, *cudaPathLengthPtr);
        glDrawArrays(GL_POINTS, 0, *cudaPathLengthPtr);
        pm.exit(PERF_OVERLAY_RENDER);
    }
    pm.exit(PERF_RENDER);
}

cudaSurfaceObject_t Renderer::createSurface() {
    CUDA_SAFE(cudaGraphicsSubResourceGetMappedArray(&cudaSurfaceDesc.res.array.array, cudaSurfaceRes, 0, 0));
    cudaSurfaceObject_t surface;
    CUDA_SAFE(cudaCreateSurfaceObject(&surface, &cudaSurfaceDesc));
    return surface;
}

void Renderer::initKernels(std::string_view cudaCode) {
    auto funcs = compiler.Compile(cudaCode, "runtime.cu", {"kernel", "genFixedPointPath"}, mode);
    kernel = funcs[0];
    pathKernel = funcs[1];
}

std::string Renderer::getPerformanceReport() {
    return pm.generateReports();
}

void Renderer::resize(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;

    CUDA_SAFE(cudaFree(cudaBuffer));
    CUDA_SAFE(cudaGraphicsUnregisterResource(cudaSurfaceRes));
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &texture);

    initTexture();
    initCuda(false);
}

int Renderer::generatePath(const std::complex<float>& z, float metricArg, const std::complex<float>& p) {
    pm.enter(PERF_OVERLAY_GEN);
    lastP = p;
    float tolerance = (mode.argIsTolerance) ? metricArg : DEFAULT_PATH_TOLERANCE;
    lastTolerance = tolerance;
    pathStart = z;

    float re = z.real();
    float im = z.imag();
    float pre = p.real();
    float pim = p.imag();
    tolerance *= tolerance;
    int maxPathSteps = MAX_PATH_STEPS; //cuLaunchKernel doesn't take const pointers so we have to make a non-const copy

    void* bufferPtr;
    CUDA_SAFE(cudaGraphicsMapResources(1, &cudaBufferRes));
    CUDA_SAFE(cudaGraphicsResourceGetMappedPointer(&bufferPtr, nullptr, cudaBufferRes));
    void* args[] = {&re, &im, &maxPathSteps, &tolerance, &bufferPtr, &cudaPathLengthPtr, &pre, &pim};
    CUDA_SAFE(cuLaunchKernel(pathKernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    CUDA_SAFE(cudaDeviceSynchronize());
    CUDA_SAFE(cudaGraphicsUnmapResources(1, &cudaBufferRes));
    pathEnabled = true; //todo add a key to remove the line
    pm.exit(PERF_OVERLAY_GEN);
    return *cudaPathLengthPtr;
}

void Renderer::refreshPathIfNeeded(const std::complex<float>& p, float metricArg) {
    assert(pathEnabled);
    float tolerance = (mode.argIsTolerance) ? metricArg : DEFAULT_PATH_TOLERANCE;
    if(std::abs(lastP - p) > PATH_PARAM_UPDATE_THRESHOLD ||
       std::abs(lastTolerance - tolerance) > PATH_TOL_UPDATE_THRESHOLD) {
        lastP = p;
        lastTolerance = tolerance;
        generatePath(pathStart, tolerance, p);
    }
}

std::pair<dist_t, dist_t> interleavedMinmax(const dist_t* buffer, size_t size) {
    dist_t min = std::numeric_limits<dist_t>::max();
    dist_t max = std::numeric_limits<dist_t>::lowest();
    for(int i = 0; i < size; i += 2) {
        if(buffer[i] < min)
            min = buffer[i];
        if(buffer[i + 1] > max)
            max = buffer[i + 1];
    }
    return {min, max};
}