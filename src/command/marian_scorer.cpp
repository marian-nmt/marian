#include "marian.h"

#include "models/model_task.h"
#include "rescorer/rescorer.h"
#include "training/graph_group.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, ConfigMode::rescoring);

  boost::timer::cpu_timer timer;
  // @TODO: support multi-gpu rescoring
  New<Rescore<Rescorer>>(options)->run();
  LOG(info, "Total time: {}", timer.format());

  return 0;
}
