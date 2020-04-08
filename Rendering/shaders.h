#pragma once
#include <string>

const std::string mainVertexShaderCode {
    #include "shaders/main_vertex.glsl"
};

const std::string mainFragmentShaderCode {
    #include "shaders/main_frag.glsl"
};

const std::string overlayVertexShaderCode {
    #include "shaders/overlay_vertex.glsl"
};

const std::string overlayFragmentShaderCode {
    #include "shaders/overlay_frag.glsl"
};