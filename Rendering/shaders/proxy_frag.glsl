R"(
#version 460 core

in vec2 vTexCoords;

uniform sampler2D tex;

out vec4 fragColor;

void main() {
    fragColor = texture(tex, vec2(vTexCoords.x, 1.0 - vTexCoords.y));
}

)"
