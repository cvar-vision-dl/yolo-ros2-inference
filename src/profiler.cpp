#include "yolo_inference_cpp/profiler.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace yolo_inference {

Profiler::ScopedTimer::ScopedTimer(Profiler& profiler, const std::string& name)
    : profiler_(profiler), name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
}

Profiler::ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start_).count();
    profiler_.timers_[name_].times.push_back(duration);

    // Keep only recent samples
    if (profiler_.timers_[name_].times.size() > TimerData::MAX_SAMPLES) {
        profiler_.timers_[name_].times.erase(profiler_.timers_[name_].times.begin());
    }
}

void Profiler::startTimer(const std::string& name) {
    timers_[name].start_time = std::chrono::high_resolution_clock::now();
}

void Profiler::endTimer(const std::string& name) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(
        end - timers_[name].start_time).count();

    timers_[name].times.push_back(duration);

    // Keep only recent samples
    if (timers_[name].times.size() > TimerData::MAX_SAMPLES) {
        timers_[name].times.erase(timers_[name].times.begin());
    }
}

double Profiler::getLastTime(const std::string& name) const {
    auto it = timers_.find(name);
    if (it != timers_.end() && !it->second.times.empty()) {
        return it->second.times.back();
    }
    return 0.0;
}

double Profiler::getAverageTime(const std::string& name) const {
    auto it = timers_.find(name);
    if (it != timers_.end() && !it->second.times.empty()) {
        double sum = std::accumulate(it->second.times.begin(), it->second.times.end(), 0.0);
        return sum / it->second.times.size();
    }
    return 0.0;
}

void Profiler::reset() {
    timers_.clear();
}

void Profiler::logStats() const {
    std::cout << "\n=== Performance Statistics ===\n";
    std::cout << std::fixed << std::setprecision(2);

    for (const auto& [name, data] : timers_) {
        if (!data.times.empty()) {
            double avg = std::accumulate(data.times.begin(), data.times.end(), 0.0) / data.times.size();
            double min_val = *std::min_element(data.times.begin(), data.times.end());
            double max_val = *std::max_element(data.times.begin(), data.times.end());

            std::cout << name << ":\n";
            std::cout << "  Avg: " << avg << "ms, Min: " << min_val << "ms, Max: " << max_val << "ms\n";
        }
    }
    std::cout << "==============================\n" << std::endl;
}