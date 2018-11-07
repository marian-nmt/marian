#include "marian.h"

#include "models/model_task.h"
#include "rescorer/rescorer.h"
#include "common/timer.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = parseOptions(argc, argv, cli::mode::scoring);

  timer::Timer timer;
  New<Rescore<Rescorer>>(options)->run();
  LOG(info, "Total time: {}", timer.format());

  return 0;
}
