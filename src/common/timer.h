#pragma once

#include <boost/timer/timer.hpp>

namespace marian {
namespace timer {
  using Timer = boost::timer::cpu_timer;
  using AutoTimer = boost::timer::auto_cpu_timer;
}
}
