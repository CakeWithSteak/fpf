R"(
#version 460 core

in float index;
out vec4 fragColor;

const vec4 lineColor = vec4(0.05, 0.2, 0.8, 1);

void main() {
    fragColor = lineColor + vec4(0, index / 10, 0, 0);
}

)"