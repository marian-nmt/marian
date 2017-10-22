#include "logging.h"
#include "common/config.h"
#include "spdlog/sinks/null_sink.h"

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
  else if(level == "err" or level == "error")
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
    validLogs.push_back(options->get<std::string>("log"));
  }

  if(options && options->has("valid-log")) {
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
}
