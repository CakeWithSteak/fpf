#pragma once
#include <vector>
#include <condition_variable>
#include <mutex>
#include <optional>

// Single producer multiple consumer queue, which can only be filled once
template <typename T>
class ProducerConsumerQueue {
    std::vector<T> items;
    size_t maxSize;
    size_t produced = 0;
    size_t consumed = 0;
    std::condition_variable cv;
    std::mutex mutex;
public:
    explicit ProducerConsumerQueue(size_t maxSize) : items(maxSize), maxSize(maxSize) {}

    void add(const T& item) {
        if(filled())
            throw std::out_of_range("Exceeded maximum size of ProducerConsumerQueue");
        items[produced] = std::move(item);
        {
            std::unique_lock lock(mutex);
            ++produced;
        }
        if(produced == maxSize)
            cv.notify_all(); //Wake all consumers once the task is complete so that everything can shut down properly
        else
            cv.notify_one();
    }

    //Returns the consumed value, or nothing if the queue is finished
    std::optional<T> consume() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this]{return produced > consumed || finished();});
        if(finished())
            return {};
        return items[consumed++];
    }

    bool finished() {
        return consumed >= maxSize;
    }

    bool filled() {
        return produced >= maxSize;
    }

    //Indicates the no more items will be produced
    void close() {
        if(!filled()) {
            maxSize = produced;
            cv.notify_all();
        }
    }
};


