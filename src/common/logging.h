#pragma once

#include "spdlog/spdlog.h"

#define LOG(logger) checkedLog(#logger)

typedef std::shared_ptr<spdlog::logger> Logger;
Logger stderrLogger(const std::string&,
                    const std::string&,
                    const std::vector<std::string>& = {});

namespace marian {
class Config;
}

// template <class... Args>
Logger checkedLog(std::string logger);
void createLoggers(const marian::Config* options = nullptr);
