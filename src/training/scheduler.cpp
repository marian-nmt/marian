#include "scheduler.h"
#include <signal.h>
#include <cassert>

namespace marian {

// SIGNAL HANDLING, see scheduler.cpp for definitions
// Currently, only the following is handled by a custom signal handler:
// SIGTERM: When SIGTERM is received, the global (static member) flag sigterm_ (false by default) is set to true
//     by signalHandler(). When sigterm_ is true, keepGoing() returns false, and the current state of training models
//     is saved prior to exiting.
//        This functionality is helpful when training on clusters with time limits on compute slots, e.g., on s
//     clusters managed by slurm. Slurm can be asked to sending a (custom) warning signal to a process at a given
//     point in time prior to the hard "time's up".

bool sigterm_{false}; // flag signalling that SIGTERM has been received false by default, set to true by signalHandler(SIGTERM)

void signalHandler(int sig) {
  // Note: sys_siglist[sig] or stdsignal() describe the effect (e.g.,
  // 'Terminated' rather than provide the signal name (which are #define(s)
  // in signal.h), so we have to do custom log messages here.
  switch (sig) {
    case SIGTERM: // save models and exit
      LOG(info, "[training] Scheduler received signal SIGTERM"); // @TODO: figure out if this is safe. The logs are global and thread-safe, so should be OK?
      sigterm_ = true;
      break;
    default:
      ABORT("No action defined for signal {}", sig);
  }
}

// installs signalHandler() for select signals (currently only SIGTERM)
void installSignalHandlers() {
  // TODO: use sigaction instead of signal, 
  // cf. https://stackoverflow.com/questions/231912/what-is-the-difference-between-sigaction-and-signal
  signal(SIGTERM, signalHandler);
}

bool getSigtermFlag() {
  return sigterm_;
}

}
