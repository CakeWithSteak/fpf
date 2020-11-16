#include "glad/glad.h"
#include "Renderer.h"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <driver_types.h>
#include <algorithm>
#include "../Computation/safeCall.h"
#include "utils.h"
#include "../Computation/shared_types.h"

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
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

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

    glBufferData(GL_ARRAY_BUFFER, 2 * std::max(MAX_PATH_STEPS, SHAPE_TRANS_DEFAULT_POINTS) * sizeof(double), nullptr, GL_DYNAMIC_DRAW);

    //Line vertices
    glVertexAttribLPointer(0, 2, GL_DOUBLE, sizeof(double) * 2, nullptr);
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

    initTextures();
    initShaders();
    initCuda();
    initKernels(cudaCode);
}

void Renderer::initTextures(bool setupProxy) {
    unsigned int textures[2];
    glGenTextures(setupProxy ? 2 : 1, textures);
    distTexture = textures[0];

    if(setupProxy) {
        proxyTexture = textures[1];

        //Init proxy texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, proxyTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, proxyTexture, 0);
    }

    //Init dist texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, distTexture);

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
    if(mode.staticMinMax.has_value()) {
        mainShader.setUniform(minimumUniform, mode.staticMinMax->first);
        mainShader.setUniform(maximumUniform, mode.staticMinMax->second);
    }

    overlayShader.use();
    viewCenterUniform = overlayShader.getUniformLocation("viewportCenter");
    viewBreadthUniform = overlayShader.getUniformLocation("viewportBreadth");

    proxyShader.use();
    proxyShader.setUniform("tex", 1);
}

void Renderer::initCuda(bool registerPathRes) {
    CUDA_SAFE(cudaSetDevice(0));
    numBlocks = ceilDivide(width * height, 1024);
    CUDA_SAFE(cudaMallocManaged(&cudaBuffer, 2 * numBlocks * sizeof(float))); //Buffer for min/max fpdist values
    CUDA_SAFE(cudaGraphicsGLRegisterImage(&cudaSurfaceRes, distTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    cudaSurfaceDesc.resType = cudaResourceTypeArray;

    if(mode.isAttractor) {
        size_t numAttractors = static_cast<int>(width * ATTRACTOR_RESOLUTION_MULT) * static_cast<int>(height * ATTRACTOR_RESOLUTION_MULT);
        CUDA_SAFE(cudaMalloc(&attractorsDeviceBuffer, numAttractors * sizeof(HostFloatComplex)));
        attractorsHostBuffer = std::make_unique<HostFloatComplex[]>(numAttractors);
    }

    if(registerPathRes) {
        CUDA_SAFE(cudaMallocManaged(&cudaPathLengthPtr, sizeof(int)));
        CUDA_SAFE(cudaGraphicsGLRegisterBuffer(&overlayBufferRes, overlayLineVBO, cudaGraphicsRegisterFlagsWriteDiscard));
    }
}


Renderer::~Renderer() {
    CUDA_SAFE(cudaGraphicsUnregisterResource(cudaSurfaceRes));
    CUDA_SAFE(cudaGraphicsUnregisterResource(overlayBufferRes));
    CUDA_SAFE(cudaFree(cudaBuffer));
    CUDA_SAFE(cudaFree(cudaPathLengthPtr));
    CUDA_SAFE(cudaFree(attractorsDeviceBuffer));
}

void Renderer::render(int maxIters, double metricArg, const std::complex<double>& p, float colorCutoff) {
    pm.enter(PERF_RENDER);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    auto [start, end] = viewport.getCorners();

    size_t numAttractors = (mode.isAttractor) ? findAttractors(maxIters, metricArg, p) : 0;

    CUDA_SAFE(cudaGraphicsMapResources(1, &cudaSurfaceRes));
    auto surface = createSurface();

    pm.enter(PERF_KERNEL);
    if(doublePrec)
        launch_kernel_double(kernel, start.real(), end.real(), start.imag(), end.imag(), maxIters, cudaBuffer, surface, width, height, p.real(), p.imag(), mode.prepMetricArg(metricArg), attractorsDeviceBuffer, numAttractors);
    else
        launch_kernel_float(kernel, start.real(), end.real(), start.imag(), end.imag(), maxIters, cudaBuffer, surface, width, height, p.real(), p.imag(), mode.prepMetricArg(metricArg), attractorsDeviceBuffer, numAttractors);
    CUDA_SAFE(cudaDeviceSynchronize());
    pm.exit(PERF_KERNEL);

    CUDA_SAFE(cudaDestroySurfaceObject(surface));
    CUDA_SAFE(cudaGraphicsUnmapResources(1, &cudaSurfaceRes));

    mainShader.use();
    if(!mode.staticMinMax.has_value()) {
        auto [min, max] = (mode.isAttractor) ? std::make_pair(0.0f, static_cast<float>(numAttractors)) : interleavedMinmax(cudaBuffer, 2 * numBlocks);
        mainShader.setUniform(minimumUniform, min);
        mainShader.setUniform(maximumUniform, std::min(max, colorCutoff));
        std::cout << "Min: " << min << " max: " << max << "\n";
    }

    glBindVertexArray(mainVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    if(isOverlayActive()) {
        refreshOverlayIfNeeded(p, metricArg);
        pm.enter(PERF_OVERLAY_RENDER);
        glBindVertexArray(overlayVAO);
        overlayShader.use();
        overlayShader.setUniform(viewCenterUniform, viewport.getCenter().real(), viewport.getCenter().imag());
        overlayShader.setUniform(viewBreadthUniform, viewport.getBreadth());
        glDrawArrays(GL_POINTS, 0, getOverlayLength());
        if(connectOverlayPoints)
            glDrawArrays(GL_LINE_STRIP, 0, getOverlayLength());
        pm.exit(PERF_OVERLAY_RENDER);
    }

    pm.exit(PERF_RENDER);
}

void Renderer::paint() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(mainVAO);
    proxyShader.use();
    glDrawArrays(GL_TRIANGLES, 0, 6);
}


cudaSurfaceObject_t Renderer::createSurface() {
    CUDA_SAFE(cudaGraphicsSubResourceGetMappedArray(&cudaSurfaceDesc.res.array.array, cudaSurfaceRes, 0, 0));
    cudaSurfaceObject_t surface;
    CUDA_SAFE(cudaCreateSurfaceObject(&surface, &cudaSurfaceDesc));
    return surface;
}

void Renderer::initKernels(std::string_view cudaCode) {
    auto funcs = compiler.Compile(cudaCode, "runtime.cu", {"kernel", "genFixedPointPath", "transformShape", "findAttractors"}, mode, doublePrec);
    kernel = funcs[0];
    pathKernel = funcs[1];
    shapeTransformKernel = funcs[2];
    findAttractorsKernel = funcs[3];
}

std::string Renderer::getPerformanceReport() {
    return pm.generateReports();
}

void Renderer::resize(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;

    CUDA_SAFE(cudaFree(cudaBuffer));
    CUDA_SAFE(cudaFree(attractorsDeviceBuffer));
    CUDA_SAFE(cudaGraphicsUnregisterResource(cudaSurfaceRes));
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &distTexture);
    glActiveTexture(GL_TEXTURE1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, newWidth, newHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    initTextures(false);
    initCuda(false);
}

int Renderer::generatePath(const std::complex<double>& z, double metricArg, const std::complex<double>& p) {
    pm.enter(PERF_OVERLAY_GEN);
    lastP = p;
    double tolerance = (mode.argIsTolerance) ? metricArg : DEFAULT_PATH_TOLERANCE;
    lastTolerance = tolerance;
    pathStart = z;

    void* bufferPtr;
    CUDA_SAFE(cudaGraphicsMapResources(1, &overlayBufferRes));
    CUDA_SAFE(cudaGraphicsResourceGetMappedPointer(&bufferPtr, nullptr, overlayBufferRes));

    if(doublePrec) {
        launch_kernel_generic(pathKernel, 1, 1, z.real(), z.imag(), MAX_PATH_STEPS, tolerance * tolerance, bufferPtr, cudaPathLengthPtr, p.real(), p.imag());
    } else {
        std::complex<float> fz(z), fp(p);
        launch_kernel_generic(pathKernel, 1, 1, fz.real(), fz.imag(), MAX_PATH_STEPS, static_cast<float>(tolerance * tolerance), bufferPtr, cudaPathLengthPtr, fp.real(), fp.imag());
    }

    CUDA_SAFE(cudaDeviceSynchronize());
    CUDA_SAFE(cudaGraphicsUnmapResources(1, &overlayBufferRes));
    pathEnabled = true;
    pm.exit(PERF_OVERLAY_GEN);
    return *cudaPathLengthPtr;
}

void Renderer::refreshOverlayIfNeeded(const std::complex<double>& p, double metricArg) {
    double tolerance = (mode.argIsTolerance) ? metricArg : DEFAULT_PATH_TOLERANCE;
    if(std::abs(lastP - p) > PATH_PARAM_UPDATE_THRESHOLD ||
      (std::abs(lastTolerance - tolerance) > PATH_TOL_UPDATE_THRESHOLD && pathEnabled)) {
        lastP = p;
        lastTolerance = tolerance;
        if(pathEnabled)
            generatePath(pathStart, tolerance, p);
        else if(shapeTransEnabled)
            generateShapeTransformImpl(p);
    }
}

void Renderer::hideOverlay() {
    pathEnabled = false;
    shapeTransEnabled = false;
}

std::vector<unsigned char> Renderer::exportImageData() {
    pm.enter(PERF_FRAME_EXPORT_TIME);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    long size = width * height * 3;
    std::vector<unsigned char> data(size);
    glReadnPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, size, data.data());
    pm.exit(PERF_FRAME_EXPORT_TIME);
    return data;
}

void Renderer::generateShapeTransform(ShapeProps shape, int iteration, const std::complex<double>& p, int numShapePoints) {
    shapeTransProps = shape;
    shapeTransIteration = iteration;
    shapeTransEnabled = true;
    //Make sure we don't overflow the overlay VBO
    if(numShapePoints > SHAPE_TRANS_DEFAULT_POINTS) {
        std::cerr << "ERROR: Number of shape transform points can't exceed " << SHAPE_TRANS_DEFAULT_POINTS << "." << std::endl;
        numShapePoints = SHAPE_TRANS_DEFAULT_POINTS;
    }
    shapeTransNumPoints = (numShapePoints == -1) ? SHAPE_TRANS_DEFAULT_POINTS : numShapePoints;
    generateShapeTransformImpl(p);
}

void Renderer::setShapeTransformIteration(int iteration, const std::complex<double>& p, bool disableIncremental) {
    auto lastIterations = shapeTransIteration;
    shapeTransIteration = iteration;

    if(!disableIncremental)
        generateShapeTransformImpl(p, lastIterations);
    else
        generateShapeTransformImpl(p);
}


void Renderer::generateShapeTransformImpl(const std::complex<double>& p, int lastIterations) {
    pm.enter(PERF_LINE_TRANS_GEN);
    constexpr int BLOCK_SIZE = 1024;
    lastP = p;

    int itersToDo;
    bool incremental = false;
    if(lastIterations == -1 || shapeTransIteration < lastIterations || mode.capturing) {
        itersToDo = shapeTransIteration;
    } else {
        itersToDo = shapeTransIteration - lastIterations;
        incremental = true;
    }

    void* bufferPtr;
    CUDA_SAFE(cudaGraphicsMapResources(1, &overlayBufferRes));
    CUDA_SAFE(cudaGraphicsResourceGetMappedPointer(&bufferPtr, nullptr, overlayBufferRes));

    if(doublePrec) {
        launch_kernel_generic(shapeTransformKernel, shapeTransNumPoints, BLOCK_SIZE, *shapeTransProps, p.real(), p.imag(), shapeTransNumPoints, itersToDo, incremental, bufferPtr);

    } else {
        std::complex<float> fp(p);
        launch_kernel_generic(shapeTransformKernel, shapeTransNumPoints, BLOCK_SIZE, *shapeTransProps, fp.real(), fp.imag(), shapeTransNumPoints, itersToDo, incremental, bufferPtr);
    }

    CUDA_SAFE(cudaDeviceSynchronize());
    CUDA_SAFE(cudaGraphicsUnmapResources(1, &overlayBufferRes));
    pm.exit(PERF_LINE_TRANS_GEN);
}

int Renderer::getOverlayLength() {
    if(pathEnabled)
        return *cudaPathLengthPtr;
    else if(shapeTransEnabled)
        return shapeTransNumPoints;
    else
        throw std::runtime_error("getOverlayLength called without overlay active");
}

void Renderer::togglePointConnections() {
    connectOverlayPoints = !connectOverlayPoints;
}

size_t Renderer::findAttractors(int maxIters, double metricArg, const std::complex<double>& p) {
    constexpr int BLOCK_SIZE = 256;
    pm.enter(PERF_ATTRACTOR);
    int aWidth = width * ATTRACTOR_RESOLUTION_MULT;
    int aHeight = height * ATTRACTOR_RESOLUTION_MULT;
    size_t bufSize = aWidth * aHeight;
    auto [start, end] = viewport.getCorners();
    auto tolerance = (mode.argIsTolerance) ? metricArg : 0.05f;

    if(doublePrec) {
        launch_kernel_generic(findAttractorsKernel, bufSize, BLOCK_SIZE, start.real(), end.real(), start.imag(), end.imag(), maxIters, p.real(), p.imag(), tolerance * tolerance, aWidth, aHeight, attractorsDeviceBuffer, ATTRACTOR_MATCH_TOL);
    } else {
        std::complex<float> fstart(start), fend(end), fp(p);
        float ftol = tolerance, F_ATTRACTOR_MATCH_TOL = ATTRACTOR_MATCH_TOL;
        launch_kernel_generic(findAttractorsKernel, bufSize, BLOCK_SIZE, fstart.real(), fend.real(), fstart.imag(), fend.imag(), maxIters, fp.real(), fp.imag(), ftol * ftol, aWidth, aHeight, attractorsDeviceBuffer, F_ATTRACTOR_MATCH_TOL);
    }

    CUDA_SAFE(cudaDeviceSynchronize());

    CUDA_SAFE(cudaMemcpy(attractorsHostBuffer.get(), attractorsDeviceBuffer, bufSize * sizeof(HostFloatComplex), cudaMemcpyDeviceToHost));
    auto res = deduplicateWithTol(attractorsHostBuffer.get(), aWidth * aHeight, ATTRACTOR_MATCH_TOL, MAX_ATTRACTORS);
    CUDA_SAFE(cudaMemcpy(attractorsDeviceBuffer, attractorsHostBuffer.get(), res * sizeof(HostFloatComplex), cudaMemcpyHostToDevice));

    std::cout << "Attractors: " << res;
    if(res == MAX_ATTRACTORS)
        std::cout << " (max)";
    std::cout << std::endl;

    pm.exit(PERF_ATTRACTOR);
    return res;
}