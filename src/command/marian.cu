#include "marian.h"

#include "models/model_task.h"
#include "training/graph_group_async.h"
#include "training/graph_group_sync.h"
#include "training/graph_group_singleton.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  auto devices = options->get<std::vector<size_t>>("devices");

  if(devices.size() > 1) {
    if(options->get<bool>("sync"))
      New<Train<SyncGraphGroup>>(options)->run();
    else
      New<Train<AsyncGraphGroup>>(options)->run();
  } else
    New<Train<SingletonGraph>>(options)->run();

  return 0;
}
