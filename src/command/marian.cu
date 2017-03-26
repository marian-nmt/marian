
#include "marian.h"
#include "models/dl4mt.h"
#include "models/gnmt.h"
#include "models/multi_gnmt.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);;
  auto devices = options->get<std::vector<size_t>>("devices");
  auto type = options->get<std::string>("type");
  
  if(devices.size() > 1) {
    if(type == "gnmt")
      Train<AsyncGraphGroup<GNMT>>(options);
    else if(type == "multi-gnmt")
      Train<AsyncGraphGroup<MultiGNMT>>(options);
    else
      Train<AsyncGraphGroup<DL4MT>>(options);    
  }
  else {
    if(type == "gnmt")
      Train<Singleton<GNMT>>(options);
    else if(type == "multi-gnmt")
      Train<Singleton<MultiGNMT>>(options);
    else
      Train<Singleton<DL4MT>>(options);
  }
  
  
  return 0;
}
