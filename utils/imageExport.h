#pragma once
#include <filesystem>

void exportImage(const std::filesystem::path& filename, int width, int height, const std::vector<unsigned char>& data);