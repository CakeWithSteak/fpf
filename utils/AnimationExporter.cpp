#include <fstream>
#include "AnimationExporter.h"
#include "imageExport.h"


//Determines the appropriate basename for the exported frames
AnimationExporter::AnimationExporter(const AnimationExporter::path& dirname, int width, int height, size_t totalFrames)
 : totalFrames(totalFrames), saveQueue(totalFrames), frames(totalFrames), width(width), height(height)
{
    const std::string prefix = "anim";
    if(!std::filesystem::exists(dirname)) {
        std::filesystem::create_directory(dirname);
        basename = dirname / (prefix + "_0");
    } else { //Check if some animations have already been exported into this directory, if yes export the new frames with an incremented id
        int highestAnimId = -1;
        for(auto& p : std::filesystem::directory_iterator(dirname)) {
            auto name = p.path().filename().string();
            if(name.starts_with(prefix)) {
                auto firstSep = name.find_first_of('_');
                if(firstSep == std::string::npos) continue;
                auto secondSep = name.find('_', firstSep + 1);
                if(secondSep == std::string::npos) continue;
                int animId = std::stoi(name.substr(firstSep + 1, secondSep));
                if(animId > highestAnimId)
                    highestAnimId = animId;
            }
        }
        basename = dirname / (prefix + "_" + ((highestAnimId != -1) ? std::to_string(highestAnimId + 1) : "0"));
    }

    initThreads();
}

void AnimationExporter::initThreads() {
    int numThreads = std::min(std::thread::hardware_concurrency() - 1, MAX_SAVE_THREADS);
    saveThreads.reserve(numThreads);
    for(int i = 0; i < numThreads; ++i) {
        saveThreads.emplace_back(&AnimationExporter::threadFunc, this);
    }
}

void AnimationExporter::joinThreads() {
    for(auto& t : saveThreads)
        t.join();
}

void AnimationExporter::saveFrame(AnimationExporter::Frame&& frame) {
    if(currFrame == totalFrames)
        throw std::out_of_range("Exceeded maximum animation frames in AnimationExporter");
    frames[currFrame] = std::move(frame);
    saveQueue.add({currFrame, &frames[currFrame]});
    ++currFrame;
}

void AnimationExporter::threadFunc() {
    while(auto item = saveQueue.consume()) {
        auto [frameIndex, frameData] = *item;
        auto filename = basename;
        filename += '_' + std::to_string(frameIndex) + ".png";
        if(std::filesystem::exists(filename))
            throw std::runtime_error("Animation export would overwrite existing file. Aborting.");
        exportImage(filename, width, height, *frameData);
        *frameData = std::vector<unsigned char>();
    }
}

AnimationExporter::~AnimationExporter() {
    joinThreads();
}

bool AnimationExporter::filled() {
    return saveQueue.filled();
}

void AnimationExporter::writeAnimReferenceString(std::string_view str) {
    const std::string filename = "animrefs.txt";
    std::ofstream file(basename.parent_path() / filename, std::ios::app);
    file << str;
    file.close();
}
