#pragma once
#include <string>

class Shader {
    unsigned int program;
public:
    Shader(std::string vertexSrc, std::string fragSrc);
    void use();
    void setUniform(const std::string& name, int val);
    void setUniform(const std::string& name, float val);
    void setUniform(int location, int val);
    void setUniform(int location, float val);
    void setUniform(int location, float v1, float v2);
    int getUniformLocation(const std::string& name);
};


