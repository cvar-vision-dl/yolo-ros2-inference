// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolo_inference
{

class Profiler
{
public:
  class ScopedTimer
  {
public:
    ScopedTimer(Profiler & profiler, const std::string & name);
    ~ScopedTimer();

private:
    Profiler & profiler_;
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
  };

  void startTimer(const std::string & name);
  void endTimer(const std::string & name);
  double getLastTime(const std::string & name) const;
  double getAverageTime(const std::string & name) const;
  void reset();
  void logStats() const;

  // Helper macro-like function
  ScopedTimer scopedTimer(const std::string & name)
  {
    return ScopedTimer(*this, name);
  }

private:
  struct TimerData
  {
    std::vector<double> times;
    std::chrono::high_resolution_clock::time_point start_time;
    static constexpr size_t MAX_SAMPLES = 100;
  };

  std::unordered_map<std::string, TimerData> timers_;
};

}  // namespace yolo_inference
