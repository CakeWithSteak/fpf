R"(
#version 460 core

in float index;
out vec4 fragColor;

const vec4 lineColorStart = vec4(0.05, 0.2, 0.8, 1);
const vec4 lineColorEnd = vec4(0.25, 1, 0.9, 1);
const float mixCoeff = 10;
const double arrowLength = 0.1; // Adds helpful "arrows" pointing towards the next step of the path.

void main() {
    int flatIndex = int(index);
    if(abs(index - flatIndex) < arrowLength) {
        fragColor = (flatIndex % 2 == 0) ? vec4(1, 0, 0.3, 1) : vec4(0.3, 0, 1, 1);
    } else {
        fragColor = mix(lineColorStart, lineColorEnd, (index / mixCoeff));
    }
}

)"