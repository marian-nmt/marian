#include "scheduler.h"
#include <signal.h>
#include <cassert>

namespace marian {
bool Scheduler::sigterm_{false};

void Scheduler::signalHandler(int sig) {
  // Note: sys_siglist[sig] or stdsignal() describe the effect (e.g.,
  // 'Terminated' rather than provide the signal name (which are #define(s)
  // in signal.h), so we have to do custom log messages here.
  switch (sig) {
    case SIGTERM: // save models and exit
      LOG(info, "[training] Scheduler received signal SIGTERM");
      Scheduler::sigterm_ = true;
      break;
    default:
      ABORT("No action defined for signal {}", sig);
  }
}


void Scheduler::installSignalHandlers() {
  // TODO: use sigaction instead of signal
  signal(SIGTERM, Scheduler::signalHandler);
}

}
