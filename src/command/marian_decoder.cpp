#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, ConfigMode::translating);

  auto task = New<TranslateMultiGPU<BeamSearch>>(options);

  boost::timer::cpu_timer timer;
  task->run();
  LOG(info)->info("Total time: {}", timer.format());

  return 0;
}
