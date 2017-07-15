#include "marian.h"
#include "rescorer/rescorer.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, ConfigMode::rescoring);
  
  boost::timer::cpu_timer timer;
  WrapModelType<Rescore, Rescorer>(options)->run();
  LOG(info)->info("Total time: {}", timer.format());

  return 0;
}
