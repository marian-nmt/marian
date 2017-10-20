#pragma once

#include "spdlog/spdlog.h"

/**
 * @brief Prints logging message into stderr and a file specified with `--log`
 * option.
 *
 * Example usage: `LOG(info, "[data] Vocab size: {}", vocabSize)`
 *
 * A good practise is to put `[namespace]` at the beginning of your message.
 *
 * @param level Logging level: trace, debug, info, warn, error, critical
 * @param ...
 */
#define LOG(level, ...) checkedLog("general", #level, __VA_ARGS__)

/**
 * @brief Prints logging message regarding validation into stderr and a file
 * specified with `--valid-log` option.
 *
 * The message is automatically preceded by "[valid] ".
 *
 * @see \def LOG(level, ...)
 */
#define LOG_VALID(level, ...) checkedLog("valid", #level, __VA_ARGS__)

typedef std::shared_ptr<spdlog::logger> Logger;
Logger stderrLogger(const std::string&,
                    const std::string&,
                    const std::vector<std::string>& = {},
                    bool quiet = false);

namespace marian {
class Config;
}

template <class... Args>
void checkedLog(std::string logger, std::string level, Args... args) {
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
