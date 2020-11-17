#pragma once
#include <filesystem>
#include <vector>
#include <thread>
#include "ProducerConsumerQueue.h"

class AnimationExporter {
    static constexpr unsigned int MAX_SAVE_THREADS = 6;
    using path = std::filesystem::path;
    using Frame = std::vector<unsigned char>;

    path basename;
    size_t totalFrames;
    size_t currFrame = 0;
    int width;
    int height;
    ProducerConsumerQueue<std::pair<size_t, Frame*>> saveQueue;
    std::vector<std::thread> saveThreads;
    std::vector<Frame> frames;

    void initThreads();
    void joinThreads();
    void threadFunc();
public:
    AnimationExporter(const path& dirname, int width, int height, size_t totalFrames);
    ~AnimationExporter();
    void saveFrame(Frame&& frame);
    bool filled();
    void stop(); //Saves frames that have already been submitted, then shuts down.
    void writeAnimReferenceString(std::string_view str);
};