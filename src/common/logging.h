#pragma once

#include "spdlog/spdlog.h"
//#include "blaze/util/logging/LogLevel.h"

namespace amunmt {

#define LOG(logger,...) spdlog::get(#logger)->info(__VA_ARGS__)

}
