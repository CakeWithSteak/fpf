#include <vector>
#include <fstream>
#include "imageExport.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

void flipImage(std::vector<unsigned char>& data, int width, int height) {
    const int c = 3;
    for(int y = 0; y < height / 2; ++y) {
        for(int x = 0; x < width; ++x) {
            int iTop = c * ((y * width) + x);
            int iBottom = c * (((height - y - 1) * width) + x);
            std::swap(data[iTop], data[iBottom]);
            std::swap(data[iTop + 1], data[iBottom + 1]);
            std::swap(data[iTop + 2], data[iBottom + 2]);
        }
    }
}

void exportImage(const std::filesystem::path& filename, int width, int height, std::vector<unsigned char>& data) {
    flipImage(data, width, height);
    stbi_write_png(filename.string().c_str(), width, height, 3, data.data(), 0);
}

std::string getReferenceString(const path& imagePath, const State& state) {
    auto [corner1, corner2] = state.viewport.getCorners();
    std::stringstream ss;
    ss << imagePath.filename().string()
        << "\t" << state.expr
        << "\t" << state.mode.displayName
        << "\t" << corner1
        << "\t" << corner2
        << "\t" << state.p
        << "\t" << state.maxIters
        << "\t" << state.metricArg
        << "\t" << (state.colorCutoffEnabled ? "Enabled" : "Disabled")
        << "\t" << state.colorCutoff << "\t";

    ss << (state.doublePrec ? "On\t" : "Off\t");

    if(state.pathStart.has_value())
        ss << state.pathStart.value();
    else
        ss << "None";

    if(state.shapeTransProps.has_value()) {
        if(state.shapeTransProps->shape == LINE) {
            ss << "\tLine" << "\t" << state.shapeTransIteration << "\t"
               << state.shapeTransProps->line.p1.std() << "\t" << state.shapeTransProps->line.p2.std();
        } else { //CIRCLE
            if(state.shapeTransNumPointsOverride == -1)
                ss << "\tCircle";
            else
                ss << "\tPolygon(" << state.shapeTransNumPointsOverride - 1 << ")";
            ss << "\t" << state.shapeTransIteration << "\t"
               << state.shapeTransProps->circle.center.std() << "\t" << state.shapeTransProps->circle.r;
        }
    } else {
        ss << "\tNone\t0\tNone\tNone";
    }

    ss << "\n";

    return ss.str();
}


void writeImageInfoToReferences(const path& refsPath, const path& imagePath, const State& state) {
    if(std::filesystem::exists(refsPath)) {
        auto str = getReferenceString(imagePath, state);
        std::ofstream refs(refsPath, std::ios::app);
        refs << str;
        refs.close();
    }
}
