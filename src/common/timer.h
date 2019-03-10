#pragma once

#ifdef _GNUC_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include <boost/timer/timer.hpp>
#ifdef _GNUC_
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
// (needed on Windows only to resolve a link error, but causes a warning on Linux)
#include <boost/chrono.hpp>
#endif

#include <chrono>
#include <sstream>

namespace marian {
namespace timer {

// Helper function to get the current date and time
static inline std::string currentDate() {
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char date[100] = {0};
  std::strftime(date, sizeof(date), "%F %X %z", std::localtime(&now));
  return date;
}

// Timer measures elapsed time.
// This is a wrapper around std::chrono providing wall time only
class Timer {
protected:
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;
  using duration = std::chrono::nanoseconds;

  time_point start_;     // Starting time point
  bool stopped_{false};  // Indicator if the timer has been stopped
  duration time_;        // Time duration from start() to stop()

public:
  // Create and start the timer
  Timer() : start_(clock::now()) {}

  // Restart the timer. It is not resuming
  void start() {
    stopped_ = false;
    start_ = clock::now();
  }

  // Stop the timer
  void stop() {
    if(stopped_)
      return;
    stopped_ = true;
    time_ = clock::now() - start_;
  }

  // Check if the timer has been stopped
  bool stopped() const { return stopped_; }

  // Get the time elapsed without stopping the timer.
  // If the template type is not specified, it returns the time counts as represented by
  // std::chrono::seconds
  template <class Duration = std::chrono::seconds>
  double elapsed() const {
    using duration_double = std::chrono::duration<double, typename Duration::period>;
    if(stopped_)
      return std::chrono::duration_cast<duration_double>(time_).count();
    return std::chrono::duration_cast<duration_double>(clock::now() - start_).count();
  }

  // Default desctructor
  virtual ~Timer() {}
};

// Automatic timer displays timing information on the standard output stream when it is destroyed
class AutoTimer : public Timer {
public:
  // Create and start the timer
  AutoTimer() : Timer() {}

  // Stop the timer and display time elapsed on std::cout
  ~AutoTimer() {
    stop();
    std::cout << "Time: " << elapsed() << "s wall" << std::endl;
  }
};

// @TODO: replace with a timer providing CPU/thread time on both Linux and Windows. This is required
// for auto-tuner.
// Check get_clocktime on Linux: https://linux.die.net/man/3/clock_gettime
// Check GetThreadTimes on Windows:
// https://docs.microsoft.com/en-gb/windows/desktop/api/processthreadsapi/nf-processthreadsapi-getthreadtimes
using CPUTimer = boost::timer::cpu_timer;

}  // namespace timer
}  // namespace marian
