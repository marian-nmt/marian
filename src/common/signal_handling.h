#pragma once
#include <csignal>
#include <string>

// SIGNAL HANDLING

// The signal handlers (and checkers) here are implemented in line with with the recommendations
// for signal handling in the SEI CERT C Coding Standard, specifically
//
// - SIG30-C:
//   https://wiki.sei.cmu.edu/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers
//
// - SIG31-C:
//   https://wiki.sei.cmu.edu/confluence/display/c/SIG31-C.+Do+not+access+shared+objects+in+signal+handlers
//
// The exact behavior of 'graceful exit' depends on the application; for training, it means 'save model and exit',
// for a server (not implemented yet): 'block new requests but serve pending requests and then exit'.
//
// Graceful exit for training is useful for training on clusters with time limits on jobs. Slurm, for example, can be
// set up to send a custom signal at a set time before the end of the time slot, giving Marian time to save its current
// state before getting killed.

namespace marian {


/// Request graceful exit (signal handler)
void requestSaveAndExit(int sig);

/// Check if graceful exit was requested.
bool saveAndExitRequested();

/// General purpose signal handler that simply sets a flag when a signal is received.
//  (only for SIGNAL No. < 32).
void setSignalFlag(int sig);  // custom handler (set flag) for sig

/// Check if a setSignalFlag was triggered for this signal
bool getSignalFlag(int sig);

} // End of namespace marian
