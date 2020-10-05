#include "PerformanceMonitor.h"
#include <chrono>
#include <algorithm>
#include <sstream>
#include <numeric>

inline uint64_t getTime() {
    auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(t).count();
}

size_t PerformanceMonitor::createCategory(std::string_view name) {
    categories.emplace_back(categories.size(), name);
    samples.emplace_back();
    samples.back().reserve(SAMPLES_RESERVE_SIZE);
    activeStartTimes.push_back(0);
    return categories.size() - 1;
}

void PerformanceMonitor::enter(size_t category) {
    activeStartTimes[category] = getTime();
}

void PerformanceMonitor::exit(size_t category) {
    auto time = getTime();
    samples[category].push_back(time - activeStartTimes[category]);
}

std::string PerformanceMonitor::generateReports() {
    std::stringstream ss;
    ss << "---------------------\n";

    for(int i =0; i < categories.size(); ++i) {
        auto& s = samples[i];
        if(s.empty())
            continue;

        std::sort(s.begin(), s.end());
        auto wholeSum = std::accumulate(s.cbegin(), s.cend(), 0ull);
        double wholeAvg = wholeSum / static_cast<double>(s.size());

        auto it10 = s.cbegin() + (9 * s.size() / 10);
        auto sum10 = std::accumulate(it10, s.cend(), 0ull);
        double avg10 = sum10 / static_cast<double>(std::distance(it10, s.cend()));

        auto it1 = s.cbegin() + (99 * s.size() / 100);
        auto sum1 = std::accumulate(it1, s.cend(), 0ull);
        double avg1 = sum1 / static_cast<double>(std::distance(it1, s.cend()));

        auto worst = s.back();

        if(i != 0)
            ss << "\n";

        ss << categories[i].first + 1 << ". \"" << categories[i].second << "\"\n"
           << "Num. samples: " << s.size() << "\n"
           << "Average time: " << wholeAvg / 1000.0 << " ms\n"
           << "Worst 10% time: " << avg10 / 1000.0 << " ms\n"
           << "Worst 1% time: " << avg1 / 1000.0 << " ms\n"
           << "Worst time: " << worst / 1000.0 << " ms\n";
    }
    ss << "---------------------\n\n";
    return ss.str();
}
