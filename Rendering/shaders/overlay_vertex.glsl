R"(
#version 460 core

layout (location = 0) in vec2 position;

uniform dvec2 viewportCenter;
uniform double viewportBreadth;

out float index;

vec2 viewportToGL(vec2 p) {
    return vec2((p - viewportCenter) / viewportBreadth); //Cast to float vec
}

void main() {
    gl_Position = vec4(viewportToGL(position), 0, 1);
    index = gl_VertexID;
}

)"