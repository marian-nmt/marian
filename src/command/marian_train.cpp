#include <signal.h>
#include "marian.h"

#include "training/graph_group_async.h"
#include "training/graph_group_multinode_sync.h"
#include "training/graph_group_singleton.h"
#include "training/graph_group_sync.h"
#include "training/training.h"

#ifdef CUDA_FOUND
#include "training/graph_group_async_drop.h"
#include "training/graph_group_multinode.h"
#endif

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

    if(options->get<bool>("sync-sgd")) {
      LOG(info, "Using multi-node synchronous training");
      New<Train<MultiNodeGraphGroupSync>>(options)->run();
    } else {
#ifdef CUDA_FOUND
      LOG(info, "Using multi-node asynchronous training");
      New<Train<MultiNodeGraphGroup>>(options)->run();
#else
      ABORT("Asynchronous multi-node training requires CUDA");
#endif
    }
  }
  // --sync-sgd always selects SyncGraphGroup
  //
  // If given, then this implementation is used for all combinations of (single, multiple) MPI
  // processes x (single, multiple) GPUs per MPI process.  This variant is presently up-to-date and
  // best supported.
  else if (options->get<bool>("sync-sgd")) {
    LOG(info, "Using synchronous training");
    New<Train<SyncGraphGroup>>(options)->run();
  }
  else {
    auto devices = Config::getDevices(options);
    if(devices.size() == 1) {
      LOG(info, "Using single-device training");
      New<Train<SingletonGraph>>(options)->run();
    } else {
      if(options->get<float>("grad-dropping-rate") > 0.0) {
#ifdef CUDA_FOUND
        LOG(info, "Using asynchronous training with gradient dropping");
        New<Train<AsyncGraphGroupDrop>>(options)->run();
#else
        ABORT("Asynchronous training with gradient dropping requires CUDA");
#endif
      } else {
        LOG(info, "Using asynchronous training");
        New<Train<AsyncGraphGroup>>(options)->run();
      }
    }
  }

  // If we exit due to SIGTERM, exit with 128 + the signal number, as suggested
  // for bash in http://tldp.org/LDP/abs/html/exitcodes.html. This allows parent
  // scripts to determine if training terminated naturally or via SIGTERM.
  // Whith this approach we can accommodate additional signals in the future.
  // An alternative would be to return 124, which is what the timeout command
  // returns for timeout -s SIGTERM <seconds> ...., because exiting after SIGTERM
  // is not technically a fatal error (which is what the 128+x convention usually
  // stands for).
  return getSigtermFlag() ? (128 + SIGTERM) : 0;
}
