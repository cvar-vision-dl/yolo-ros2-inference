#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolo_inference {

class Profiler {
public:
    class ScopedTimer {
    public:
        ScopedTimer(Profiler& profiler, const std::string& name);
        ~ScopedTimer();
    private:
        Profiler& profiler_;
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };

    void startTimer(const std::string& name);
    void endTimer(const std::string& name);
    double getLastTime(const std::string& name) const;
    double getAverageTime(const std::string& name) const;
    void reset();
    void logStats() const;

    // Helper macro-like function
    ScopedTimer scopedTimer(const std::string& name) {
        return ScopedTimer(*this, name);
    }

private:
    struct TimerData {
        std::vector<double> times;
        std::chrono::high_resolution_clock::time_point start_time;
        static constexpr size_t MAX_SAMPLES = 100;
    };

    std::unordered_map<std::string, TimerData> timers_;
};

} // namespace yolo_inference
