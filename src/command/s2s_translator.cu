#include "marian.h"
#include "translator/translator.h"
#include "translator/beam_search.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, true, true);
  WrapModelType<Translate, BeamSearch>(options)->run();
  
  return 0;

}
