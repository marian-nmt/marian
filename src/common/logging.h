#pragma once

#include "spdlog/spdlog.h"

#define LOG(logger, ...) checkedLog(#logger, __VA_ARGS__)

typedef std::shared_ptr<spdlog::logger> Logger;
Logger stderrLogger(const std::string&,
                    const std::string&,
                    const std::vector<std::string>& = {});

namespace marian {
class Config;
}

template <class... Args>
void checkedLog(std::string logger, Args... args) {
  if(spdlog::get(logger))
    spdlog::get(logger)->info(args...);
}

void createLoggers(const marian::Config* options = nullptr);
