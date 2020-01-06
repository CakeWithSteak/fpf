R"(
#version 460 core

in vec2 vTexCoords;

uniform isampler2D distances;
uniform int minDist;
uniform int maxDist;

out vec4 fragColor;

const float maxHue = 0.8f; //Pink

//http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float mapHue(float distance) {
    if(minDist == maxDist) return 0;

    //Linear
    //return (distance - minDist) / float(maxDist - minDist);

    //Better linear
    return (distance - minDist) * (maxHue/(maxDist - minDist));

    //Everything is red
    //return 0.8;

    //1 is red
    //return distance == 1 ? 0 : 0.5;
}

void main() {
    int distance = texture(distances, vTexCoords).r;
    if(distance == -1) {
        fragColor = vec4(0, 0, 0, 1);
    } else {
        float hue = mapHue(distance);
        vec3 rgb = hsv2rgb(vec3(hue, 1, 1));
        fragColor = vec4(rgb, 1);
    }
}

)"
