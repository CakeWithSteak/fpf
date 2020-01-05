cmake_minimum_required(VERSION 3.15)
project(fixed_point_fractals LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-8)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -rdynamic -g")

add_executable(fixed_point_fractals main.cpp Rendering/glad/glad.c Computation/kernel.cu Computation/kernel.h Rendering/Window.cpp Rendering/Window.h Rendering/Renderer.cpp Rendering/Renderer.h Rendering/shaders.h Computation/cudaSafeCall.h utils/Viewport.cpp utils/Viewport.h utils/Timer.cpp utils/Timer.h)

target_include_directories(fixed_point_fractals PRIVATE /usr/local/cuda/include)

target_compile_options(fixed_point_fractals PRIVATE -O3)

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