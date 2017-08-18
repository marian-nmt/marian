#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, ConfigMode::translating);
  auto task = New<TranslateLoopMultiGPU<BeamSearch>>(options);

  boost::timer::cpu_timer timer;

  for(std::string line; std::getline(std::cin, line);) {
    timer.start();
    for(auto& output : task->run({line}))
      std::cout << output << std::endl;
    LOG(info)->info("Search took: {}", timer.format(5, "%ws"));
  }

  return 0;
}
