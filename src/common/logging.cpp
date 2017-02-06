#include "logging.h"

std::shared_ptr<spdlog::logger> stderrLogger(const std::string& name,
                                             const std::string& pattern) {
  auto logger = spdlog::stderr_logger_mt(name);
  logger->set_pattern(pattern);
  return logger;
}

