#include "logging.h"
#include "common/config.h"
#include "spdlog/sinks/null_sink.h"
#include <time.h>
#include <stdlib.h>

std::shared_ptr<spdlog::logger> stderrLogger(
    const std::string& name,
    const std::string& pattern,
    const std::vector<std::string>& files,
    bool quiet) {
  std::vector<spdlog::sink_ptr> sinks;

  auto stderr_sink = spdlog::sinks::stderr_sink_mt::instance();

  if(!quiet)
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

bool setLoggingLevel(spdlog::logger& logger, std::string const level) {
  if(level == "trace")
    logger.set_level(spdlog::level::trace);
  else if(level == "debug")
    logger.set_level(spdlog::level::debug);
  else if(level == "info")
    logger.set_level(spdlog::level::info);
  else if(level == "warn")
    logger.set_level(spdlog::level::warn);
  else if(level == "err" || level == "error")
    logger.set_level(spdlog::level::err);
  else if(level == "critical")
    logger.set_level(spdlog::level::critical);
  else if(level == "off")
    logger.set_level(spdlog::level::off);
  else {
    logger.warn("Unknown log level '{}' for logger '{}'",
                level.c_str(),
                logger.name().c_str());
    return false;
  }
  return true;
}

void createLoggers(const marian::Config* options) {
  std::vector<std::string> generalLogs;
  std::vector<std::string> validLogs;

  if(options && options->has("log")) {
    generalLogs.push_back(options->get<std::string>("log"));
#ifndef _WIN32
    // can't open the same file twice in Windows for some reason
    validLogs.push_back(options->get<std::string>("log"));
#endif
  }

  if(options && options->has("valid-log")
     && !options->get<std::string>("valid-log").empty()) {
    validLogs.push_back(options->get<std::string>("valid-log"));
  }

  bool quiet = options && options->get<bool>("quiet");
  Logger general{
      stderrLogger("general", "[%Y-%m-%d %T] %v", generalLogs, quiet)};
  Logger valid{
      stderrLogger("valid", "[%Y-%m-%d %T] [valid] %v", validLogs, quiet)};

  if(options && options->has("log-level")) {
    std::string loglevel = options->get<std::string>("log-level");
    if(!setLoggingLevel(*general, loglevel))
      return;
    setLoggingLevel(*valid, loglevel);
  }

  if (options && options->has("log-time-zone")) {
      std::string timezone = options->get<std::string>("log-time-zone");
      if (timezone != "") {
#ifdef _WIN32
#define setenv(var, val, over) SetEnvironmentVariableA(var, val) // ignoring over flag
#endif
        setenv("TZ", timezone.c_str(), true);
        tzset();
      }
  }
}
