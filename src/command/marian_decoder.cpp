#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#ifdef _WIN32
#include <Windows.h>
#endif

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, ConfigMode::translating);
  auto task = New<TranslateMultiGPU<BeamSearch>>(options);

  boost::timer::cpu_timer timer;
  task->run();
  LOG(info, "Total time: {}", timer.format());

#ifdef _WIN32 // debug CRT throws an error that free() is used for an aligned allocation; disable this error for now
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif

  return 0;
}
