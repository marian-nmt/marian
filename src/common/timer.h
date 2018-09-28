#pragma once

#include <boost/timer/timer.hpp>

#include <chrono>
#include <sstream>

namespace marian {
namespace timer {

class Timer {
protected:
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;
  using duration = std::chrono::nanoseconds;

  time_point start_;
  duration time_;
  bool stopped_{false};

public:
  Timer() : start_(clock::now()) {}

  void start() {
    stopped_ = false;
    start_ = clock::now();
  }

  void stop() {
    if(stopped_)
      return;
    stopped_ = true;
    time_point current = clock::now();
    time_ = current - start_;
  }

  bool stopped() const { return stopped_; }

  std::string format(size_t precision = 5, const std::string& fmt = "") const {
    auto seconds = std::chrono::duration<double>(elapsed()).count();
    auto format = "%." + std::to_string(precision) + "g";
    char buffer[50];
    std::snprintf(buffer, 50, format.c_str(), seconds);
    return buffer;
  }

  std::chrono::nanoseconds elapsed() const {
    if(stopped_)
      return time_;
    time_point current = clock::now();
    return current - start_;
  }

  virtual ~Timer() {}
};

using CPUTimer = boost::timer::cpu_timer;
using AutoTimer = boost::timer::auto_cpu_timer;

}  // namespace timer
}  // namespace marian
