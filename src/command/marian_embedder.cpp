#include "marian.h"

#include "models/model_task.h"
#include "embedder/embedder.h"
#include "common/timer.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = parseOptions(argc, argv, cli::mode::embedding);
  New<Embed<Embedder>>(options)->run();
  
  return 0;
}
