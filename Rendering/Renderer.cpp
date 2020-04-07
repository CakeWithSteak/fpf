#include "glad/glad.h"
#include "Renderer.h"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <driver_types.h>
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
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    //Position
    glVertexAttribPointer(0, 2, GL_FLOAT, false, sizeof(float) * 4, nullptr);
    glEnableVertexAttribArray(0);

    //UV
    glVertexAttribPointer(1, 2, GL_FLOAT, false, sizeof(float) * 4, (void*)(sizeof(float) * 2));
    glEnableVertexAttribArray(1);

    initTexture();
    initShaders();
    initCuda();
    initKernel(cudaCode);
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
    shader.use();

    shader.setUniform("distances", 0);
    shader.setUniform("maxHue", mode.maxHue);

    minimumUniform = shader.getUniformLocation("minDist");
    maximumUniform = shader.getUniformLocation("maxDist");

    if(mode.staticMinMax.has_value()) {
        shader.setUniform(minimumUniform, mode.staticMinMax->first);
        shader.setUniform(maximumUniform, mode.staticMinMax->second);
    }
}

void Renderer::initCuda() {
    CUDA_SAFE(cudaSetDevice(0));
    numBlocks = (width * height + 1024 - 1) / 1024; //Ceiled division
    CUDA_SAFE(cudaMallocManaged(&cudaBuffer, 2 * numBlocks * sizeof(int))); //Buffer for min/max fpdist values
    CUDA_SAFE(cudaGraphicsGLRegisterImage(&cudaSurfaceRes, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    cudaSurfaceDesc.resType = cudaResourceTypeArray;
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

    if(!mode.staticMinMax.has_value()) {
        auto [min, max] = interleavedMinmax(cudaBuffer, 2 * numBlocks);
        shader.setUniform(minimumUniform, min);
        shader.setUniform(maximumUniform, std::min(max, colorCutoff));
        std::cout << "Min: " << min << " max: " << max << "\n";
    }

    glDrawArrays(GL_TRIANGLES, 0, 6);
    pm.exit(PERF_RENDER);
}

cudaSurfaceObject_t Renderer::createSurface() {
    CUDA_SAFE(cudaGraphicsSubResourceGetMappedArray(&cudaSurfaceDesc.res.array.array, cudaSurfaceRes, 0, 0));
    cudaSurfaceObject_t surface;
    CUDA_SAFE(cudaCreateSurfaceObject(&surface, &cudaSurfaceDesc));
    return surface;
}

void Renderer::initKernel(std::string_view cudaCode) {
    kernel = compiler.Compile(cudaCode, "runtime.cu", "kernel", mode);
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
    initCuda();
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