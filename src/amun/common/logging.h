#pragma once

#include "spdlog/spdlog.h"
#include <string>

namespace amunmt {

#define LOG(logger) spdlog::get(#logger)
void set_loglevel(spdlog::logger& logger, std::string const level);
  
// #define LOG(logger,...) spdlog::get(#logger)->info(__VA_ARGS__)
}
