#pragma once

#include <chrono>
#include <memory>

class Timer {
public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  void tic() { start_time_ = Clock::now(); }

  template <typename T = std::chrono::microseconds> double toc() const {
    auto elapsed_time = Clock::now() - start_time_;
    return std::chrono::duration_cast<T>(elapsed_time).count();
  }

private:
  TimePoint start_time_;
};