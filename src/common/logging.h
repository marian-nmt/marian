#pragma once

#include "spdlog/spdlog.h"

#define LOG(logger) spdlog::get(#logger)->info()
