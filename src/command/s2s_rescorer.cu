#include "marian.h"
#include "training/rescorer.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, true, false);

  // @TODO: these options should be set in relevant classes
  options->set<bool>("dynamic-batching", false);
  options->set<size_t>("maxi-batch", 1);
  options->set<size_t>("max-length", 1000);

  auto task = rescorerByType(options);

  boost::timer::cpu_timer timer;
  task->run();
  LOG(info, "Total time: {}", timer.format());

  return 0;
}
