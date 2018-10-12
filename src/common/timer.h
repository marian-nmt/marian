#pragma once

#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp> // (needed on Windows only to resolve a link error)

namespace marian {
namespace timer {
  using Timer = boost::timer::cpu_timer;
  using AutoTimer = boost::timer::auto_cpu_timer;
}
}
