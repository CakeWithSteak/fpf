R"(
#version 460 core

layout (location = 0) in vec2 position;

uniform vec2 viewportCenter;
uniform float viewportBreadth;

out float index;

vec2 viewportToGL(vec2 p) {
    return (p - viewportCenter) / viewportBreadth;
}

void main() {
    gl_Position = vec4(viewportToGL(position), 0, 1);
    index = gl_VertexID;
}

)"