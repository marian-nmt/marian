#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "batch_generator.h"
#include "optimizers.h"
#include "trainer.h"
#include "models/mlp.h"

using namespace marian;
using namespace keywords;
using namespace data;
using namespace models;

int main(int argc, char** argv) {

  auto mlp = FeedforwardClassifier({784, 2048, 2048, 10});
  mlp.graphviz("mnist_benchmark.dot");

  /*****************************************************/

  auto train = DataSet<MNIST>("../examples/mnist/train-images-idx3-ubyte",
                              "../examples/mnist/train-labels-idx1-ubyte");

  Train(mlp, train,
        optimizer=Optimizer<Adam>(0.0002),
        batch_size=200,
        max_epochs=50);

  mlp.dump("mnist.mrn");

  return 0;
}
