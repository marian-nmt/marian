#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "examples/mnist/mnist.h"
#include "examples/mnist/trainer.h"

using namespace marian;
using namespace data;
using namespace keywords;


int main(int argc, char** argv) {
  auto options = New<Config>(argc, argv, false);
  auto device = options->get<std::vector<size_t>>("devices").front();

  auto trainSet =
    DataSet<MNIST>("../src/examples/mnist/train-images-idx3-ubyte",
                   "../src/examples/mnist/train-labels-idx1-ubyte");
  auto validSet =
    DataSet<MNIST>("../src/examples/mnist/t10k-images-idx3-ubyte",
                   "../src/examples/mnist/t10k-labels-idx1-ubyte");

  std::vector<int> networkDims = {trainSet->dim(0), 2048, 2048, 10};

  {
    auto ff = New<ExpressionGraph>();
    ff->setDevice(device);
    ff->reserveWorkspaceMB(512);

    auto validator =
      Run<Tester<MNIST>>(ff, networkDims, validSet,
                         batch_size=200);

    auto trainer =
      Run<Trainer<MNIST>>(ff, networkDims, trainSet,
                          optimizer=Optimizer<Adam>(0.002),
                          valid=validator,
                          batch_size=200,
                          max_epochs=10);
      trainer->run();

      //validator->run();
  }

  return 0;
}
