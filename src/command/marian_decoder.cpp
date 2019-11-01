#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "common/timer.h"
#ifdef _WIN32
#include <Windows.h>
#endif

int main(int argc, char** argv) {
  using namespace marian;
  auto options = parseOptions(argc, argv, cli::mode::translation);
  auto task = New<Translate<BeamSearch>>(options);

  timer::Timer timer;
  task->run();
  LOG(info, "Total time: {:.5f}s wall", timer.elapsed());

  return 0;
}
