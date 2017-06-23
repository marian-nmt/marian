#include "logging.h"

namespace amunmt {

void set_loglevel(spdlog::logger& logger, std::string const level) {
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
  else
    logger.warn("Unknown log level '{}' for logger '{}'",
		level.c_str(), logger.name().c_str());
}
  
}

