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

bool configureMPI(int, char**, bool);

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  auto devices = options->getDevices();

  if(options->get<bool>("multi-node")) {
    LOG(warn, "[experimental] Running multi-node training");

    if(options->get<bool>("sync-sgd")) {
      New<Train<MultiNodeGraphGroupSync>>(options)->run();
    } else {
#ifdef CUDA_FOUND
      New<Train<MultiNodeGraphGroup>>(options)->run();
#else
      ABORT("Asynchronous multi-node training requires CUDA");
#endif
    }
  } else {
    if(devices.size() == 1) {
      New<Train<SingletonGraph>>(options)->run();
    } else {
      if(options->get<bool>("sync-sgd")) {
        New<Train<SyncGraphGroup>>(options)->run();
      } else if(options->get<float>("grad-dropping-rate") > 0.0) {
#ifdef CUDA_FOUND
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
