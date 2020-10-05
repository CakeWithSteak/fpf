#pragma once
#include <complex>
#include <memory>
#include "../Computation/kernel.h"
#include "../utils/Viewport.h"
#include "../Compilation/NvrtcCompiler.h"
#include "../utils/PerformanceMonitor.h"
#include "../modes.h"
#include "Shader.h"
#include "shaders.h"
#include "utils.h"
#include "../Computation/constants.h"
#include "../Computation/shared_types.h"

class Renderer {
    int width;
    int height;
    const Viewport& viewport;
    ModeInfo mode;
    bool doublePrec;

    bool connectOverlayPoints = true;

    bool pathEnabled = false;
    std::complex<double> pathStart;
    std::complex<double> lastP;
    double lastTolerance;

    bool shapeTransEnabled = false;
    std::optional<ShapeProps> shapeTransProps;
    int shapeTransIteration;

    unsigned int numBlocks;
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
    cudaGraphicsResource_t overlayBufferRes = nullptr;
    cudaResourceDesc cudaSurfaceDesc;
    float* cudaBuffer = nullptr;
    int* cudaPathLengthPtr = nullptr;
    HostFloatComplex* attractorsDeviceBuffer = nullptr;
    std::unique_ptr<HostFloatComplex[]> attractorsHostBuffer = nullptr;

    NvrtcCompiler compiler;
    CUfunction kernel;
    CUfunction pathKernel;
    CUfunction shapeTransformKernel;
    CUfunction findAttractorsKernel;

    PerformanceMonitor pm{"Main kernel time", "Total render time", "Orbit compute time", "Overlay render time", "Line transform compute time", "Attractor compute time"};
    static constexpr size_t PERF_KERNEL = 0;
    static constexpr size_t PERF_RENDER = 1;
    static constexpr size_t PERF_OVERLAY_GEN = 2;
    static constexpr size_t PERF_OVERLAY_RENDER = 3;
    static constexpr size_t PERF_LINE_TRANS_GEN = 4;
    static constexpr size_t PERF_ATTRACTOR = 5;

    static constexpr int MAX_PATH_STEPS = 1024;
    static constexpr double PATH_PARAM_UPDATE_THRESHOLD = 0.01f;
    static constexpr double PATH_TOL_UPDATE_THRESHOLD = 0.001f;
    static constexpr double DEFAULT_PATH_TOLERANCE = 0.001f;

    static constexpr int LINE_TRANS_NUM_POINTS = 500'000;

    // Not really worth it to set this below 1 -- 0.5 is surprisingly only a 3% performance increase.
    // Predictably the attractor compute time falls dramatically, but at the same time the main kernel time also grows for some reason
    // despite the detected attractors being the same.
    static constexpr double ATTRACTOR_RESOLUTION_MULT = 1.0f;
    static constexpr size_t MAX_ATTRACTORS = 32;
    static constexpr double ATTRACTOR_MATCH_TOL = KERNEL_ATTRACTOR_MAX_TOL;

    void init(std::string_view cudaCode);
    void initTexture();
    void initShaders();
    void initCuda(bool registerPathRes = true);
    void initKernels(std::string_view cudaCode);
    cudaSurfaceObject_t createSurface();
    void refreshOverlayIfNeeded(const std::complex<double>& p, double metricArg);
    void generateShapeTransformImpl(const std::complex<double>& p, int lastIterations = -1);
    inline bool isOverlayEnabled() { return pathEnabled || shapeTransEnabled; }
    int getOverlayLength();
    size_t findAttractors(int maxIters, double metricArg, const std::complex<double>& p);
public:
    Renderer(int width, int height, const Viewport& viewport, const ModeInfo& mode, std::string_view cudaCode, bool doublePrec)
            : width(width), height(height), viewport(viewport), mode(mode), doublePrec(doublePrec) {init(cudaCode);}
    ~Renderer();

    void render(int maxIters, double metricArg, const std::complex<double>& p, float colorCutoff);
    std::string getPerformanceReport();
    void resize(int newWidth, int newHeight);
    int generatePath(const std::complex<double>& z, double tolerance, const std::complex<double>& p);
    void hideOverlay();
    std::vector<unsigned char> exportImageData();
    void generateShapeTransform(ShapeProps shape, int iteration, const std::complex<double>& p);
    void setShapeTransformIteration(int iteration, const std::complex<double>& p, bool disableIncremental = false);
    void togglePointConnections();
};


