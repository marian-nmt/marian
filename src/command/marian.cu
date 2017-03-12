
#include "marian.h"
#include "models/gnmt.h"
#include "models/dl4mt2.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);;

  if(options->get<std::string>("type") == "dl4mt")
    Train<AsyncGraphGroup<DL4MT>>(options);
  else
    Train<AsyncGraphGroup<GNMT>>(options);

  return 0;
}
