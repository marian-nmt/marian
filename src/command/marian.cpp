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

int main(int argc, char** argv) {
  using namespace marian;

  auto options = parseOptions(argc, argv);

  // selects MultiNodeGraphGroup family
  // Note: --sync-sgd without --multi-node also supports MPI now, using the SyncGraphGroup.
  // This means we have two redundant implementations of multi-node sync-sgd. Note that the
  // MultiNodeGraphGroup family is out of date. Therefore, the goal is to remove MultiNodeGraphGroupSync.
  if(options->get<bool>("multi-node")) {
    LOG(warn, "[experimental] Old multi-node training implementations. These are presently not up-to-date.");

    if(options->get<bool>("sync-sgd")) {
      LOG(warn, "[training] Using MultiNodeGraphGroupSync trainer.");
      New<Train<MultiNodeGraphGroupSync>>(options)->run();
    } else {
#ifdef CUDA_FOUND
      LOG(warn, "[training] Using MultiNodeGraphGroup trainer.");
      New<Train<MultiNodeGraphGroup>>(options)->run();
#else
      ABORT("Asynchronous multi-node training requires CUDA");
#endif
    }
  }
  // --sync-sgd always selects SyncGraphGroup
  // If given, then this implementation is used for all combinations of
  // (single, multiple) MPI processes x (single, multiple) GPUs per MPI process.
  // This variant is presently up-to-date and best supported.
  else if (options->get<bool>("sync-sgd")) {
    LOG(warn, "[training] Using SyncGraphGroup trainer.");
    New<Train<SyncGraphGroup>>(options)->run();
  }
  else {
    auto devices = Config::getDevices(options);
    if(devices.size() == 1) {
      LOG(warn, "[training] Using SingletonGraph trainer.");
      New<Train<SingletonGraph>>(options)->run();
    } else {
      if(options->get<float>("grad-dropping-rate") > 0.0) {
#ifdef CUDA_FOUND
        LOG(warn, "[training] Using AsyncGraphGroupDrop trainer.");
        New<Train<AsyncGraphGroupDrop>>(options)->run();
#else
        ABORT("Asynchronous training with gradient dropping requires CUDA");
#endif
      } else {
        New<Train<AsyncGraphGroup>>(options)->run();
      }
    }
  }

  return 0;
}
