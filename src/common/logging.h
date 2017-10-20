#pragma once

#include "spdlog/spdlog.h"

#define LOG(logger) checkedLog(#logger)

#define LOG2(level, ...) checkedLog2("info", #level, __VA_ARGS__)

#define LOG2_VALID(level, ...) checkedLog2("valid", #level, __VA_ARGS__)

typedef std::shared_ptr<spdlog::logger> Logger;
Logger stderrLogger(const std::string&,
                    const std::string&,
                    const std::vector<std::string>& = {},
                    bool quiet = false);

namespace marian {
class Config;
}

// TODO: remove
Logger checkedLog(std::string logger);

template <class... Args>
void checkedLog2(std::string logger, std::string level, Args... args) {
  Logger log = spdlog::get(logger);
  if(!log)
    return;

  if(level == "trace")
    log->trace(args...);
  else if(level == "debug")
    log->debug(args...);
  else if(level == "info")
    log->info(args...);
  else if(level == "warn")
    log->warn(args...);
  else if(level == "error")
    log->error(args...);
  else if(level == "critical")
    log->critical(args...);
  else {
    log->warn("Unknown log level '{}' for logger '{}'", level, logger);
  }
}

void createLoggers(const marian::Config* options = nullptr);
