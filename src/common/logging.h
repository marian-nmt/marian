#pragma once

#include "spdlog/spdlog.h"

#define LOG(logger) spdlog::get(#logger)->info()

typedef std::shared_ptr<spdlog::logger> Logger;
Logger stderrLogger(const std::string&, const std::string&);

