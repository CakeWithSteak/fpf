cmake_minimum_required(VERSION 3.15)
project(fixed_point_fractals LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-8)

add_custom_command(OUTPUT kernel.ii
        COMMAND cpp -DBUILD_FOR_NVRTC -x c++ -P -std=c++17 -I ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/Computation/kernel.cu |
        sed "1 iR\"(" |
        sed -e "\$a\)\"" > kernel.ii
        DEPENDS Computation/kernel.cu Computation/*.cuh Computation/kernel_types.h Computation/metrics.h Computation/distance_metrics/*.cuh
        IMPLICIT_DEPENDS CXX VERBATIM)

add_executable(fixed_point_fractals main.cpp Rendering/glad/glad.c Rendering/Window.cpp Rendering/Renderer.cpp utils/Viewport.cpp utils/Timer.cpp Computation/kernel_types.h Compilation/NvrtcCompiler.cpp kernel.ii Computation/dummy.cu Compilation/compileExpression.cpp Compilation/expr2cuda/expression_tree/ConstantNode.cpp Compilation/expr2cuda/expression_tree/ExpressionNode.cpp Compilation/expr2cuda/expression_tree/OperatorNode.cpp Compilation/expr2cuda/expression_tree/RootNode.cpp Compilation/expr2cuda/expression_tree/VariableNode.cpp Compilation/expr2cuda/token_transformations/infixtoprefix.cpp Compilation/expr2cuda/token_transformations/token_prep.cpp Compilation/expr2cuda/token_transformations/tokenize.cpp Compilation/expr2cuda/tree_transformations/buildTree.cpp Compilation/expr2cuda/tree_transformations/coalesceConstants.cpp Compilation/expr2cuda/tree_transformations/coalesceConstants.h Compilation/expr2cuda/expression_tree/nodeType.h utils/PerformanceMonitor.cpp utils/PerformanceMonitor.h Computation/metrics.h modes.h cli.cpp cli.h modes.cpp utils/Input.cpp utils/Input.h controls.h utils/isordered.h utils/State.h utils/serialization.cpp utils/serialization.h)

target_include_directories(fixed_point_fractals PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(fixed_point_fractals PRIVATE /usr/local/cuda/include)

target_compile_options(fixed_point_fractals PRIVATE -O3 -g)

target_link_libraries(fixed_point_fractals boost_program_options)
target_link_libraries(fixed_point_fractals boost_serialization)
target_link_libraries(fixed_point_fractals glfw)
target_link_libraries(fixed_point_fractals GL)
target_link_libraries(fixed_point_fractals pthread)
target_link_libraries(fixed_point_fractals X11)
target_link_libraries(fixed_point_fractals Xrandr)
target_link_libraries(fixed_point_fractals Xi)
target_link_libraries(fixed_point_fractals dl)
target_link_libraries(fixed_point_fractals Xinerama)
target_link_libraries(fixed_point_fractals Xxf86vm)
target_link_libraries(fixed_point_fractals Xext)
target_link_libraries(fixed_point_fractals Xcursor)
target_link_libraries(fixed_point_fractals nvrtc)
target_link_libraries(fixed_point_fractals cuda)