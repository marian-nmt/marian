#pragma once

#include <iostream>

#include "spdlog/spdlog.h"

// set to 1 to use for debugging if no loggers can be created
#define LOG_TO_STDERR 0

namespace marian {
  void logCallStack(size_t skipLevels);
  std::string getCallStack(size_t skipLevels);

  // Marian gives a basic exception guarantee. If you catch a
  // MarianRuntimeError you must assume that the object can be
  // safely destructed, but cannot be used otherwise.

  // Internal multi-threading in exception-throwing mode is not
  // allowed; and constructing a thread-pool will cause an exception.

  class MarianRuntimeException : public std::runtime_error {
  private:
    std::string callStack_;

  public:
    MarianRuntimeException(const std::string& message, const std::string& callStack)
    : std::runtime_error(message),
      callStack_(callStack) {}

    const char* getCallStack() const throw() {
      return callStack_.c_str();
    }
  };

  // Get the state of throwExceptionOnAbort (see logging.cpp), by default false
  bool getThrowExceptionOnAbort();

  // Set the state of throwExceptionOnAbort (see logging.cpp)
  void setThrowExceptionOnAbort(bool);
}

/**
 * Prints logging message into stderr and a file specified with `--log` option.
 *
 * Example usage: `LOG(info, "[data] Vocab size: {}", vocabSize)`
 *
 * A good practice is to put `[namespace]` at the beginning of the message.
 *
 * @param level Logging level: trace, debug, info, warn, error, critical
 * @param ... Message text and variables
 */
#define LOG(level, ...) checkedLog("general", #level, __VA_ARGS__)

// variant that prints the log message only upon the first time the call site is executed
#define LOG_ONCE(level, ...) do { \
  static bool logged = false;     \
  if (!logged)                    \
  {                               \
    logged = true;                \
    LOG(level, __VA_ARGS__);      \
  }                               \
} while(0)

/**
 * Prints logging message regarding validation into stderr and a file specified
 * with `--valid-log` option.
 *
 * The message is automatically preceded by "[valid] ".
 *
 * @see \def LOG(level, ...)
 */
#define LOG_VALID(level, ...) checkedLog("valid", #level, __VA_ARGS__)

// variant that prints the log message only upon the first time the call site is executed
#define LOG_VALID_ONCE(level, ...) do { \
  static bool logged = false;     \
  if (!logged)                    \
  {                               \
    logged = true;                \
    LOG_VALID(level, __VA_ARGS__);      \
  }                               \
} while(0)

#ifdef __GNUC__
#define FUNCTION_NAME __PRETTY_FUNCTION__
#else
#ifdef _WIN32
#define FUNCTION_NAME __FUNCTION__
#else
#define FUNCTION_NAME "???"
#endif
#endif

/**
 * Prints critical error message and causes abnormal program termination by
 * calling std::abort().
 *
 * @param ... Message text and variables
 */
#define ABORT(...)                                                               \
  do {                                                                           \
    auto logger = spdlog::get("general");                                        \
    if(logger == nullptr)                                                        \
      logger = createStderrLogger("general", "[%Y-%m-%d %T] Error: %v");         \
    else                                                                         \
      logger->set_pattern("[%Y-%m-%d %T] Error: %v");                            \
    checkedLog("general", "critical", __VA_ARGS__);                              \
    checkedLog("general", "critical", "Aborted from {} in {}:{}",                \
               FUNCTION_NAME, __FILE__, __LINE__);                               \
    logger->set_pattern("%v");                                                   \
    auto callStack = marian::getCallStack(/*skipLevels=*/0);                     \
    checkedLog("general", "critical", callStack);                                \
    if(marian::getThrowExceptionOnAbort())                                       \
      throw marian::MarianRuntimeException(fmt::format(__VA_ARGS__), callStack); \
    else                                                                         \
      std::abort();                                                              \
  } while(0)

/**
 * Prints critical error message and causes abnormal program termination if
 * conditions is true.
 *
 * @param condition Condition expression
 * @param ... Message text and variables
 *
 * @see \def ABORT(...)
 */
#define ABORT_IF(condition, ...) \
  do {                           \
    if(condition) {              \
      ABORT(__VA_ARGS__);        \
    }                            \
  } while(0)

#define ABORT_UNLESS(condition, ...) \
  do {                               \
    if(!(bool)(condition)) {         \
      ABORT(__VA_ARGS__);            \
    }                                \
  } while(0)

typedef std::shared_ptr<spdlog::logger> Logger;
Logger createStderrLogger(const std::string&,
                          const std::string&,
                          const std::vector<std::string>& = {},
                          bool quiet = false);

namespace marian {
class Config;
}

template <class... Args>
void checkedLog(std::string logger, std::string level, Args... args) {
#if LOG_TO_STDERR
  std::cerr << "[" << level << "] " << fmt::format(args...) << std::endl;
#else
  Logger log = spdlog::get(logger);
  if(!log) {
    return;
  }

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
#endif
}

void createLoggers(const marian::Config* options = nullptr);
void switchToMultinodeLogging(std::string nodeIdStr);
