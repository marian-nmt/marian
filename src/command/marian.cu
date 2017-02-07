
#include "marian.h"
#include "models/dl4mt.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  
  //validator = New<ValidPerplexity<DL4MT>>(options);
  //float cost = validator->validate(graph);
  
  Train<AsyncGraphGroup<DL4MT>>(options);

  return 0;
}
