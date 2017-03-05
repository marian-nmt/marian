
#include "marian.h"
#include "models/gnmt.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);;
  
  Train<AsyncGraphGroup<GNMT>>(options);

  return 0;
}
