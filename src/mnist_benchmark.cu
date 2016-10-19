#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "trainer.h"
#include "models/feedforward.h"

using namespace marian;
using namespace keywords;
using namespace data;
using namespace models;

int main(int argc, char** argv) {

  auto trainSet =
    DataSet<MNIST>("../examples/mnist/train-images-idx3-ubyte",
                   "../examples/mnist/train-labels-idx1-ubyte");

  auto validSet =
    DataSet<MNIST>("../examples/mnist/t10k-images-idx3-ubyte",
                   "../examples/mnist/t10k-labels-idx1-ubyte");

  auto ff =
    FeedforwardClassifier({
      validSet->dim(0), validSet->dim(1)
    });

  //ff->graphviz("mnist_benchmark.dot");

  auto trainer =
    Run<Trainer>(ff, trainSet,
                 optimizer=Optimizer<Adam>(0.0002),
                 batch_size=200,
                 max_epochs=50);
  trainer->run();

  auto validator =
    Run<Validator>(ff, validSet,
                   batch_size=200);
  validator->run();


  return 0;
}
