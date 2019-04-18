#include "scheduler.h"
#include <signal.h>

namespace marian {
bool Scheduler::sigterm_{false};
bool Scheduler::sigusr1_{false};
bool Scheduler::sigusr2_{false};

void
Scheduler::
signalHandler_(int sig) {
  switch (sig) {
  case SIGTERM: Scheduler::sigterm_ = true; break;
  case SIGUSR1: Scheduler::sigusr1_ = true; break;
  case SIGUSR2: Scheduler::sigusr2_ = true; break;
  default:
    ABORT("This signal handler should not have been installed for signal ",
          strsignal(sig));
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
