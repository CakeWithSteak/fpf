cmake_minimum_required(VERSION 3.17)
project(fixed_point_fractals LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

find_package(CUDAToolkit REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS program_options serialization REQUIRED)
set(GLFW_USE_STATIC_LIBS ON)
find_package(GLFW3 3.3 REQUIRED)
find_package(X11)

set(KERNEL_II_DEPS Computation/kernel.cu Computation/kernel_types.cuh Computation/metrics.cuh Computation/constants.h Computation/attractor.cuh Computation/kernel_macros.cuh Computation/kernel_types.cuh Computation/math.cuh Computation/math_float.cuh Computation/math_double.cuh Computation/shape_transform.cuh Computation/shared_types.h Computation/utils.cuh Computation/distance_metrics/fixedpoint.cuh Computation/distance_metrics/fixedpoint_euclid.cuh Computation/distance_metrics/julia.cuh Computation/distance_metrics/julia-complete.cuh Computation/distance_metrics/periodic.cuh Computation/distance_metrics/vectorfield_angle.cuh Computation/distance_metrics/vectorfield_mag.cuh)
if(UNIX)
    add_custom_command(OUTPUT kernel.ii
            COMMAND cpp -DBUILD_FOR_NVRTC -x c++ -P -std=c++2a -I ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/Computation/kernel.cu |
                    sed "1 iR\"(" |
                    sed -e "\$a\)\"" > ${CMAKE_BINARY_DIR}/kernel.ii
            DEPENDS ${KERNEL_II_DEPS}
            IMPLICIT_DEPENDS CXX VERBATIM)
elseif(MSVC)
    add_custom_command(OUTPUT kernel.ii
            COMMAND powershell -ExecutionPolicy Bypass -File ${CMAKE_SOURCE_DIR}/cmake/winbuild.ps1 -compiler ${CMAKE_CXX_COMPILER} -builddir ${CMAKE_BINARY_DIR} -srcdir ${CMAKE_SOURCE_DIR} -includes ${CUDAToolkit_INCLUDE_DIRS}
            DEPENDS ${KERNEL_II_DEPS}
            IMPLICIT_DEPENDS CXX VERBATIM)
endif()

add_executable(fixed_point_fractals main.cpp Rendering/glad/glad.c Rendering/Window.cpp Rendering/Renderer.cpp utils/Viewport.cpp utils/Timer.cpp Compilation/NvrtcCompiler.cpp kernel.ii Compilation/compileExpression.cpp Compilation/expr2cuda/expression_tree/ConstantNode.cpp Compilation/expr2cuda/expression_tree/ExpressionNode.cpp Compilation/expr2cuda/expression_tree/OperatorNode.cpp Compilation/expr2cuda/expression_tree/RootNode.cpp Compilation/expr2cuda/expression_tree/VariableNode.cpp Compilation/expr2cuda/token_transformations/infixtoprefix.cpp Compilation/expr2cuda/token_transformations/token_prep.cpp Compilation/expr2cuda/token_transformations/tokenize.cpp Compilation/expr2cuda/tree_transformations/buildTree.cpp Compilation/expr2cuda/tree_transformations/coalesceConstants.cpp Compilation/expr2cuda/tree_transformations/coalesceConstants.h Compilation/expr2cuda/expression_tree/nodeType.h utils/PerformanceMonitor.cpp utils/PerformanceMonitor.h Computation/metrics.cuh modes.h cli.cpp cli.h modes.cpp Input/Input.cpp Input/Input.h controls.h utils/State.h utils/serialization.cpp utils/serialization.h Rendering/Shader.cpp Rendering/Shader.h utils/imageExport.cpp utils/imageExport.h Rendering/utils.h Rendering/utils.cpp Computation/constants.h Input/Controller.h Input/Animator.cpp Input/Animator.h Input/AnimationParams.h utils/ProducerConsumerQueue.h utils/AnimationExporter.cpp utils/AnimationExporter.h Computation/shared_types.h utils/findCuda.h utils/findCuda.cpp utils/confirmOverwrite.h)

target_include_directories(fixed_point_fractals PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(fixed_point_fractals PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(fixed_point_fractals PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(fixed_point_fractals PRIVATE ${GLFW3_INCLUDE_DIR})

target_link_libraries(fixed_point_fractals Boost::program_options)
target_link_libraries(fixed_point_fractals Boost::serialization)
target_link_libraries(fixed_point_fractals ${GLFW3_LIBRARY})
target_link_libraries(fixed_point_fractals CUDA::cudart)
target_link_libraries(fixed_point_fractals CUDA::cuda_driver)
target_link_libraries(fixed_point_fractals CUDA::nvrtc)

if(UNIX)
    find_package(Threads)
    target_link_libraries(fixed_point_fractals Threads::Threads)
    target_link_libraries(fixed_point_fractals dl)
    target_link_libraries(fixed_point_fractals X11::X11)
    target_link_libraries(fixed_point_fractals X11::Xrandr)
    target_link_libraries(fixed_point_fractals X11::Xi)
    target_link_libraries(fixed_point_fractals X11::Xinerama)
    target_link_libraries(fixed_point_fractals X11::Xxf86vm)
    target_link_libraries(fixed_point_fractals X11::Xext)
    target_link_libraries(fixed_point_fractals X11::Xcursor)
endif()