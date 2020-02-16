cmake_minimum_required(VERSION 3.15)
project(fixed_point_fractals LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-8)

add_custom_command(OUTPUT kernel.ii
                   COMMAND cpp -DBUILD_FOR_NVRTC -x c++ -P -std=c++17 -I ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/Computation/kernel.cu |
                    sed "1 iR\"(" |
                    sed -e "\$a\)\"" > kernel.ii
                   DEPENDS Computation/kernel.cu Computation/math.cuh Computation/kernel_types.h
                   IMPLICIT_DEPENDS CXX VERBATIM)

add_executable(fixed_point_fractals main.cpp Rendering/glad/glad.c Computation/kernel.h Rendering/Window.cpp Rendering/Window.h Rendering/Renderer.cpp Rendering/Renderer.h Rendering/shaders.h Computation/safeCall.h utils/Viewport.cpp utils/Viewport.h utils/Timer.cpp utils/Timer.h Computation/kernel_types.h Compilation/NvrtcCompiler.cpp Compilation/NvrtcCompiler.h kernel.ii Computation/dummy.cu)

target_include_directories(fixed_point_fractals PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(fixed_point_fractals PRIVATE /usr/local/cuda/include)

target_compile_options(fixed_point_fractals PRIVATE -O0 -g)

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