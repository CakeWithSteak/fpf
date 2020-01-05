#include "glad/glad.h"
#include "Renderer.h"
#include "shaders.h"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <driver_types.h>
#include "../Computation/cudaSafeCall.h"


std::pair<fpdist_t, fpdist_t> interleavedMinmax(fpdist_t* buffer, size_t size);

float data[] = {
    //XY position and UV coordinates

    -1,  1,  0, 0,  //top left
    -1, -1,  0, 1, //bottom left
     1,  1,  1, 0, //top right

     1,  1,  1, 0, //top right
    -1, -1,  0, 1, //bottom left
     1, -1,  1, 1, //bottom right
};

void Renderer::init() {
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
}

void Renderer::initTexture() {
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    std::vector<fpdist_t> initialData(width*height, -1);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, width, height);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void Renderer::initShaders() {
    shaderProgram = glCreateProgram();

    unsigned int fragShader, vertShader;

    vertShader = glCreateShader(GL_VERTEX_SHADER);
    auto vertSrcPtr = vertexShaderCode.c_str();
    glShaderSource(vertShader, 1, &vertSrcPtr, nullptr);
    glCompileShader(vertShader);

    char infoLog[512];
    int success;
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertShader, sizeof(infoLog), nullptr, infoLog);
        std::string str(infoLog);
        throw std::runtime_error(("Failed to load vertex shader: " + str).c_str());
    }

    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    auto fragSrcPtr = fragmentShaderCode.c_str();
    glShaderSource(fragShader, 1, &fragSrcPtr, nullptr);
    glCompileShader(fragShader);

    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(fragShader, sizeof(infoLog), nullptr, infoLog);
        std::string str(infoLog);
        throw std::runtime_error(("Failed to load fragment shader: " + str).c_str());
    }

    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);

    glLinkProgram(shaderProgram);

    glDeleteShader(fragShader);
    glDeleteShader(vertShader);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(fragShader, sizeof(infoLog), nullptr, infoLog);
        std::string str(infoLog);
        throw std::runtime_error(("Failed to link shader program: " + str).c_str());
    }

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "distances"), 0);

    minimumUniform = glGetUniformLocation(shaderProgram, "minDist");
    maximumUniform = glGetUniformLocation(shaderProgram, "maxDist");
}

void Renderer::initCuda() {
    CUDA_SAFE(cudaSetDevice(0));
    numBlocks = (width * height) / 1024;
    if((width * height) % 1024 != 0)
        ++numBlocks;
    CUDA_SAFE(cudaMallocManaged(&cudaBuffer, 2 * numBlocks * sizeof(int)));
    CUDA_SAFE(cudaGraphicsGLRegisterImage(&cudaSurfaceRes, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    cudaSurfaceDesc.resType = cudaResourceTypeArray;
}

Renderer::~Renderer() {
    CUDA_SAFE(cudaGraphicsUnregisterResource(cudaSurfaceRes));
    CUDA_SAFE(cudaFree(cudaBuffer));
}

void Renderer::render(fpdist_t maxIters) {
    auto [start, end] = viewport.getCorners();

    CUDA_SAFE(cudaGraphicsMapResources(1, &cudaSurfaceRes));
    auto surface = createSurface();
    launch_kernel(start.real(), end.real(), start.imag(), end.imag(), tolerance, maxIters, cudaBuffer, surface, width, height);
    CUDA_SAFE(cudaDeviceSynchronize());

    CUDA_SAFE(cudaDestroySurfaceObject(surface));
    CUDA_SAFE(cudaGraphicsUnmapResources(1, &cudaSurfaceRes));

    auto [min, max] = interleavedMinmax(cudaBuffer, 2 * numBlocks);
    glUniform1i(minimumUniform, min);
    glUniform1i(maximumUniform, max);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

cudaSurfaceObject_t Renderer::createSurface() {
    CUDA_SAFE(cudaGraphicsSubResourceGetMappedArray(&cudaSurfaceDesc.res.array.array, cudaSurfaceRes, 0, 0));
    cudaSurfaceObject_t surface;
    CUDA_SAFE(cudaCreateSurfaceObject(&surface, &cudaSurfaceDesc));
    return surface;
}

std::pair<fpdist_t, fpdist_t> interleavedMinmax(fpdist_t* buffer, size_t size) {
    int min = std::numeric_limits<int>::max();
    int max = std::numeric_limits<int>::min();
    for(int i = 0; i < size; i += 2) {
        if(buffer[i] < min)
            min = buffer[i];
        if(buffer[i + 1] > max)
            max = buffer[i + 1];
    }
    return {min, max};
}