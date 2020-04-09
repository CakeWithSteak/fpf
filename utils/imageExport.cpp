#include <vector>
#include "imageExport.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

void exportImage(const std::filesystem::path& filename, int width, int height, const std::vector<unsigned char>& data) {
    stbi_write_png(filename.c_str(), width, height, 3, data.data(), 0);
}
