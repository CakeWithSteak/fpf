#include "Shader.h"
#include "glad/glad.h"
#include <stdexcept>

Shader::Shader(std::string vertexSrc, std::string fragSrc) {
    program = glCreateProgram();

    unsigned int fragShader, vertShader;

    vertShader = glCreateShader(GL_VERTEX_SHADER);
    auto vertSrcPtr = vertexSrc.c_str();
    glShaderSource(vertShader, 1, &vertSrcPtr, nullptr);
    glCompileShader(vertShader);

    char infoLog[512];
    int success;
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertShader, sizeof(infoLog), nullptr, infoLog);
        std::string str(infoLog);
        throw std::runtime_error(("Failed to load vertex shader: " + str).c_str());
    }

    fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    auto fragSrcPtr = fragSrc.c_str();
    glShaderSource(fragShader, 1, &fragSrcPtr, nullptr);
    glCompileShader(fragShader);

    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(fragShader, sizeof(infoLog), nullptr, infoLog);
        std::string str(infoLog);
        throw std::runtime_error(("Failed to load fragment shader: " + str).c_str());
    }

    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);

    glLinkProgram(program);

    glDeleteShader(fragShader);
    glDeleteShader(vertShader);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        std::string str(infoLog);
        throw std::runtime_error(("Failed to link shader program: " + str).c_str());
    }
}

void Shader::use() {
    glUseProgram(program);
}

void Shader::setUniform(const std::string& name, int val) {
    glUniform1i(getUniformLocation(name), val);
}

void Shader::setUniform(const std::string& name, float val) {
    glUniform1f(getUniformLocation(name), val);
}

int Shader::getUniformLocation(const std::string& name) {
    return glGetUniformLocation(program, name.c_str());
}

void Shader::setUniform(int location, int val) {
    glUniform1i(location, val);
}

void Shader::setUniform(int location, float val) {
    glUniform1f(location, val);
}

void Shader::setUniform(int location, float v1, float v2) {
    glUniform2f(location, v1, v2);
}
