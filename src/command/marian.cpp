#include "marian.h"

#include "training/graph_group_async.h"
#include "training/graph_group_async_drop.h"
#include "training/graph_group_singleton.h"
#include "training/graph_group_sync.h"
#include "training/training.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  auto devices = options->get<std::vector<size_t>>("devices");

  if(devices.size() > 1) {
    if(options->get<bool>("sync-sgd"))
      New<Train<SyncGraphGroup>>(options)->run();
    else if(options->get<float>("grad-dropping-rate") > 0.0)
      New<Train<AsyncGraphGroupDrop>>(options)->run();
    else
      New<Train<AsyncGraphGroup>>(options)->run();
  } else {
    New<Train<SingletonGraph>>(options)->run();
  }

  return 0;
}
