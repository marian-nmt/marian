#pragma once

#include <boost/timer/timer.hpp>
#ifdef _MSC_VER
#include <boost/chrono.hpp> // (needed on Windows only to resolve a link error, but causes a warning on Linux)
#endif

namespace marian {
namespace timer {
  using Timer = boost::timer::cpu_timer;
  using AutoTimer = boost::timer::auto_cpu_timer;
}
}
