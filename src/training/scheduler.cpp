#include "scheduler.h"
#include <signal.h>
#include <cassert>

namespace marian {
bool Scheduler::sigterm_{false};
bool Scheduler::sigusr1_{false};
bool Scheduler::sigusr2_{false};

void
Scheduler::
signalHandler_(int sig) {
  // Note: sys_siglist[sig] or stdsignal() describe the effect (e.g.,
  // 'Terminated' rather than provide the signal name (which are #define(s)
  // in signal.h), so we have to do custom log messages here.
  switch (sig) {
  case SIGTERM: // save models and exit
    LOG(info, "[training] Scheduler received signal SIGTERM");
    Scheduler::sigterm_ = true;
    break;
  case SIGUSR1: // currently has no effect
    LOG(info, "[training] Scheduler received signal SIGUSR1");
    Scheduler::sigusr1_ = true;
    break;
  case SIGUSR2: // currently has no effect
    LOG(info, "[training] Scheduler received signal SIGUSR2");
    Scheduler::sigusr2_ = true;
    break;
  default:
    ABORT("No action defined for signal {}", sig);
  }
}


void
Scheduler::
installSignalHandlers_() {
  // TODO: use sigaction instead of signal
  signal(SIGTERM, Scheduler::signalHandler_);
  signal(SIGUSR1, Scheduler::signalHandler_);
  signal(SIGUSR2, Scheduler::signalHandler_);
}

}
