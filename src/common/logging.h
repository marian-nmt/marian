#pragma once

#include "spdlog/spdlog.h"

#define LOG(logger,...) spdlog::get(#logger)->info(__VA_ARGS__)

typedef std::shared_ptr<spdlog::logger> Logger;
Logger stderrLogger(const std::string&, const std::string&,
                    const std::vector<std::string>& = {});

namespace marian {
  class Config;
}

void createLoggers(const marian::Config& options);


