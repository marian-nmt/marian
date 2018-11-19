#include "common/timer.h"
#include "common/logging.h"

#include <iostream>
#include <string>
#include <vector>

// small test program for playing around with spdlog formatting of messages

std::shared_ptr<spdlog::logger> stderrLoggerTest(
    const std::string& name,
    const std::string& pattern,
    const std::vector<std::string>& files) {
  std::vector<spdlog::sink_ptr> sinks;

  auto stderr_sink = spdlog::sinks::stderr_sink_mt::instance();
  sinks.push_back(stderr_sink);

  for(auto&& file : files) {
    auto file_sink
        = std::make_shared<spdlog::sinks::simple_file_sink_st>(file, true);
    sinks.push_back(file_sink);
  }

  auto logger
      = std::make_shared<spdlog::logger>(name, begin(sinks), end(sinks));

  spdlog::register_logger(logger);
  logger->set_pattern(pattern);
  return logger;
}

int main() {
  std::vector<std::string> logfiles;
  Logger info(stderrLoggerTest("info", "[%Y-%m-%d %T] %v", logfiles));

  info->info("hello {:06.2f}", .7);

  marian::timer::Timer timer;

  info->info("time is {:.5f} bla {:.2f}", timer.elapsed(), .7);
}
