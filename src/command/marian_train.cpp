#include <signal.h>
#include "marian.h"

#include "common/signal_handling.h"
#include "training/graph_group_async.h"
#include "training/graph_group_singleton.h"
#include "training/graph_group_sync.h"
#include "training/training.h"

#include "3rd_party/ExceptionWithCallStack.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = parseOptions(argc, argv, cli::mode::training);

  // selects MultiNodeGraphGroup family
  //
  // Note: --sync-sgd without --multi-node also supports MPI now, using the SyncGraphGroup.  This
  // means we have two redundant implementations of multi-node sync-sgd.  Note that the
  // MultiNodeGraphGroup family is out of date.  Therefore, the goal is to remove
  // MultiNodeGraphGroupSync.
  if(options->get<bool>("multi-node")) {
    LOG(warn, "[experimental] Using old multi-node training implementations that are not up-to-date");
    ABORT("Old multi-node training code disabled");
  }
  // --sync-sgd always selects SyncGraphGroup
  //
  // If given, then this implementation is used for all combinations of (single, multiple) MPI
  // processes x (single, multiple) GPUs per MPI process.  This variant is presently up-to-date and
  // best supported.
  else if (options->get<bool>("sync-sgd")) {
    LOG(info, "Using synchronous SGD");
    New<Train<SyncGraphGroup>>(options)->run();
  }
  else {
    auto devices = Config::getDevices(options);
    if(devices.size() == 1) {
      LOG(info, "[training] Using single-device training");
      New<Train<SingletonGraph>>(options)->run();
    } else {
      LOG(info, "Using asynchronous training");
      New<Train<AsyncGraphGroup>>(options)->run();
    }
  }
  // If we exit due to a graceful exit request via SIGTERM, exit with 128 + SIGTERM,
  // as suggested for bash in http://tldp.org/LDP/abs/html/exitcodes.html. This allows parent
  // scripts to determine if training terminated naturally or via SIGTERM.
  // An alternative would be to exit with code 124, which is what the timeout command
  // returns for timeout -s SIGTERM <seconds> ...., because exiting after SIGTERM
  // is not technically a fatal error (which is what the 128+x convention usually
  // stands for).
  exit(getSignalFlag(SIGTERM) ? 128 + SIGTERM : EXIT_SUCCESS);
}
