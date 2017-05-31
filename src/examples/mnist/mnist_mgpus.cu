#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <string>

#include <boost/timer/timer.hpp>

#include "marian.h"
//#include "models/model_task.h"

//#include "examples/mnist/mnist.h"
#include "examples/mnist/training.h"
#include "examples/mnist/graph_group.h"
#include "examples/mnist/mnist_model.h"


using namespace marian;
using namespace data;
using namespace keywords;


int main(int argc, char** argv) {
  auto options = New<Config>(argc, argv, false);
  auto devices = options->get<std::vector<size_t>>("devices");

  if(devices.size() > 1)
    New<MNISTTrain<MNISTAsyncGraphGroup<MNISTModel>>>(options)->run();
  else
    New<MNISTTrain<MNISTSingleton<MNISTModel>>>(options)->run();

  return 0;
}
