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
    ABORT_IF(!configureMPI(argc, argv, options->get<bool>("sync-sgd")),
             "MPI not found.");
    LOG(warn, "[experimental] Running multi-node training");

    if(options->get<bool>("sync-sgd")) {
      New<Train<MultiNodeGraphGroupSync>>(options)->run();
    }
    else {
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
      }
      else if(options->get<float>("grad-dropping-rate") > 0.0) {
#ifdef CUDA_FOUND
        New<Train<AsyncGraphGroupDrop>>(options)->run();
#else
      ABORT("Asynchronous training with gradient dropping requires CUDA");
#endif
      }
      else {
        New<Train<AsyncGraphGroup>>(options)->run();
      }
    }
  }

  return 0;
}

bool configureMPI(int argc, char** argv, bool sync) {
  bool enable = false;
#if MPI_FOUND
  int required_mode = sync ? MPI_THREAD_SERIALIZED : MPI_THREAD_MULTIPLE;
  int provided_thread_mode = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_mode);
  // Enable if occasional truncation errors
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  ABORT_IF(
      provided_thread_mode < required_mode,
      "Your version of MPI does not support multi-threaded communication.");

  enable = true;
#endif
  return enable;
}
