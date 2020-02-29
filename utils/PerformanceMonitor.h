#pragma once
#include <cstdint>
#include <vector>
#include <string>

class PerformanceMonitor {
    // Reserve a lot of memory to ensure we don't have to expand the samples vector, which would be a performance hit
    const size_t SAMPLES_RESERVE_SIZE = 2'000'000;
    std::vector<std::pair<size_t, std::string>> categories;
    std::vector<std::vector<uint64_t>> samples;
    std::vector<uint64_t> activeStartTimes;
public:
    PerformanceMonitor() = default;
    size_t createCategory(std::string_view name);
    void enter(size_t category);
    void exit(size_t category);
    std::string generateReports();

    template <class ...Ts, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<Ts, std::string_view>...>>>
    explicit PerformanceMonitor(Ts... args) {
        (createCategory(args), ...);
    }
};