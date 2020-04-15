#pragma once
#include <filesystem>
#include "State.h"

using path = std::filesystem::path;

void exportImage(const path& filename, int width, int height, const std::vector<unsigned char>& data);
void writeImageInfoToReferences(const path& referencesPath, const path& imagePath, const State& state);