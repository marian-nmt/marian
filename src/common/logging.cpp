#include "logging.h"
#include "common/config.h"
#include "spdlog/sinks/null_sink.h"

std::shared_ptr<spdlog::logger> stderrLogger(
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

bool
set_loglevel(spdlog::logger& logger, std::string const level) {
  if (level == "trace")
    logger.set_level(spdlog::level::trace);
  else if (level == "debug")
    logger.set_level(spdlog::level::debug);
  else if (level == "info")
    logger.set_level(spdlog::level::info);
  else if (level == "err" or level == "error")
    logger.set_level(spdlog::level::err);
  else if (level == "critical")
    logger.set_level(spdlog::level::critical);
  else if (level == "off")
    logger.set_level(spdlog::level::off);
  else {
    logger.warn("Unknown log level '{}' for logger '{}'",
                level.c_str(), logger.name().c_str());
    return false;
  }
  return true;
}

Logger checkedLog(std::string logger) {
  Logger ret = spdlog::get(logger);
  if(ret) {
    return ret;
  }
  else {
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_st>();
    return std::make_shared<spdlog::logger>("null_logger", null_sink);
  }
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

  Logger info{stderrLogger("info", "[%Y-%m-%d %T] %v", generalLogs)};
  Logger warn{stderrLogger("warn", "[%Y-%m-%d %T] [warn] %v", generalLogs)};
  Logger config{stderrLogger("config", "[%Y-%m-%d %T] [config] %v", generalLogs)};
  Logger memory{stderrLogger("memory", "[%Y-%m-%d %T] [memory] %v", generalLogs)};
  Logger data{stderrLogger("data", "[%Y-%m-%d %T] [data] %v", generalLogs)};
  Logger valid{stderrLogger("valid", "[%Y-%m-%d %T] [valid] %v", validLogs)};
  Logger translate{stderrLogger("translate", "%v")};
  Logger devnull{stderrLogger("devnull", "%v")};
  devnull->set_level(spdlog::level::off);

  if(options && options->has("log-level")) {
    std::string loglevel = options->get<std::string>("log-level");
    if (!set_loglevel(*info, loglevel)) return;
    set_loglevel(*warn, loglevel);
    set_loglevel(*config, loglevel);
    set_loglevel(*memory, loglevel);
    set_loglevel(*data, loglevel);
    set_loglevel(*valid, loglevel);
    set_loglevel(*translate, loglevel);
  }

}
