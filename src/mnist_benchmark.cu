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

using namespace marian;
using namespace keywords;
using namespace data;

ExpressionGraph build_graph(const std::vector<int>& dims) {
  std::cerr << "Building model... ";
  boost::timer::cpu_timer timer;

  ExpressionGraph g;
  auto x = named(g.input(shape={whatevs, dims.front()}), "x");
  auto y = named(g.input(shape={whatevs, dims.back()}), "y");

  std::vector<Expr> layers, weights, biases;
  for(int i = 0; i < dims.size()-1; ++i) {
    int in = dims[i];
    int out = dims[i+1];

    if(i == 0)
      layers.emplace_back(dropout(x, value=0.2));
    else
      layers.emplace_back(dropout(relu(dot(layers.back(), weights.back()) + biases.back()), value=0.5));

    weights.emplace_back(
      named(g.param(shape={in, out}, init=uniform()), "W" + std::to_string(i)));
    biases.emplace_back(
      named(g.param(shape={1, out}, init=zeros), "b" + std::to_string(i)));
  }

  auto scores = named(dot(layers.back(), weights.back()) + biases.back(),
                      "scores");

  auto cost = named(mean(cross_entropy(scores, y), axis=0), "cost");

  // If we uncomment the line below, this will just horribly diverge.
  // auto dummy_probs = named(softmax(scores), "dummy_probs");

  std::cerr << timer.format(5, "%ws") << std::endl;
  return g;
};


template <class DataSet, class Opt>
void Train(ExpressionGraph& g,
          DataSet& dataset,
          Opt&& opt,
          size_t batchSize,
          size_t epochs)
{
  boost::timer::cpu_timer trainTimer;
  BatchGenerator<DataSet> bg(dataset, batchSize);
  size_t update = 0;
  for(int epoch = 1; epoch <= epochs; ++epoch) {
    boost::timer::cpu_timer epochTimer;
    bg.prepare();

    float cost = 0;
    float totalExamples = 0;
    while(bg) {
      BatchPtr batch = bg.next();
      opt(g, batch);
      cost += g["cost"].val()[0] * batch->dim();
      totalExamples += batch->dim();
      update++;
    }
    cost = cost / totalExamples;

    std::cerr << "Epoch: " << std::setw(std::to_string(epochs).size())
      << epoch << "/" << epochs << " - Update: " << update
      << " - Cost: " << std::fixed << std::setprecision(4) << cost
      << " - Time: " << epochTimer.format(2, "%ws")
      << " - " << trainTimer.format(0, "%ws") << std::endl;
  }
}


int main(int argc, char** argv) {
  ExpressionGraph g = build_graph({784, 2048, 2048, 10});

  std::ofstream viz("mnist_benchmark.dot");
  viz << g.graphviz() << std::endl;
  viz.close();

  MnistDataSet mnistTrain(
   {"../examples/mnist/train-images-idx3-ubyte",
    "../examples/mnist/train-labels-idx1-ubyte"});

  Train(g, mnistTrain,
        Adam(0.0002), 200, 50);

  //Train(g, mnistTrain,
  //      optimizer=Adam(0.0002),
  //      batch_size=200,
  //      epochs=50,
  //      cost=g["cost"],
  //      output=g["scores"],
  //      displayFreq=10,
  //      holdout=0.01,
  //      holdout_metric=accuracy);


  return 0;
}
