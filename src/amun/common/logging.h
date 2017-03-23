#pragma once

#include "spdlog/spdlog.h"

namespace amunmt {

#define LOG(logger,...) spdlog::get(#logger)->info(__VA_ARGS__)

}
