#include "marian.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);;
  auto devices = options->get<std::vector<size_t>>("devices");
  
  if(devices.size() > 1)
    WrapModelType<Train, AsyncGraphGroup>(options)->run();
  else
    WrapModelType<Train, Singleton>(options)->run();
  
  return 0;
}
