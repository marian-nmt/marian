#include <algorithm>
#include <chrono>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "npz_converter.h"
#include "optimizers.h"

using namespace marian;
using namespace keywords;

const size_t IMAGE_SIZE = 784;
const size_t LABEL_SIZE = 10;
int BATCH_SIZE = 200;

ExpressionGraph build_graph(const std::vector<int>& dims) {
  std::cerr << "Building model... ";
  boost::timer::cpu_timer timer;
    
  ExpressionGraph g;
  auto x = named(g.input(shape={whatevs, IMAGE_SIZE}), "x");
  auto y = named(g.input(shape={whatevs, LABEL_SIZE}), "y");
  
  std::vector<Expr> layers, weights, biases;
  for(int i = 0; i < dims.size()-1; ++i) {
    int in = dims[i];
    int out = dims[i+1];
      
    if(i == 0) {
      layers.emplace_back(x);
    }
    else {
      layers.emplace_back(tanh(dot(layers.back(), weights.back())) + biases.back());
    }
    
    weights.emplace_back(
      g.param(shape={in, out},
            init=normal()));
    biases.emplace_back(
      g.param(shape={1, out},
            init=normal()));
  }

  Expr scores = named(dot(layers.back(), weights.back()) + biases.back(),
                      "scores");

  Expr cost = mean(cross_entropy(scores, y), axis=0);
  //auto cost = mean(-sum(y * log(softmax(scores)), axis=1), axis=0);
  Expr costreg = named(
    cost, "cost"
  );

  // If we uncomment the line below, this will just horribly diverge.
  // auto dummy_probs = named(softmax(scores), "dummy_probs");

  std::cout << g.graphviz() << std::endl;

  std::cerr << timer.format(5, "%ws") << std::endl;
  return g;
}

void shuffle(std::vector<float>& x, std::vector<float>& y, size_t dimx, size_t dimy) {
  std::srand(std::time(0));
  std::vector<size_t> ind;
  for(size_t i = 0; i < x.size() / dimx; ++i) {
    ind.push_back(i);
  }
  
  std::random_shuffle(ind.begin(), ind.end());
  
  std::vector<float> xShuffled(x.size());
  std::vector<float> yShuffled(y.size());
  
  int j = 0;
  for(auto i : ind) {
    std::copy(x.begin() + j * dimx, x.begin() + j * dimx + dimx, xShuffled.begin() + i * dimx);
    std::copy(y.begin() + j * dimy, y.begin() + j * dimy + dimy, yShuffled.begin() + i * dimy);
    j++;
  }
  
  x = xShuffled;
  y = yShuffled;
  
}

int main(int argc, char** argv) {

  int trainRows, testRows;
  
  std::cerr << "Loading train set...";
  std::vector<float> trainImages = datasets::mnist::ReadImages("../examples/mnist/train-images-idx3-ubyte", trainRows, IMAGE_SIZE);
  std::vector<float> trainLabels = datasets::mnist::ReadLabels("../examples/mnist/train-labels-idx1-ubyte", trainRows, LABEL_SIZE);
  std::cerr << "Done." << std::endl;
  
  std::cerr << "Loading test set...";
  std::vector<float> testImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", testRows, IMAGE_SIZE);
  std::vector<float> testLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", testRows, LABEL_SIZE);
  std::cerr << "Done." << std::endl;

  ExpressionGraph g = build_graph({IMAGE_SIZE, 2000, 2000, 2000, 2000, 2000, LABEL_SIZE});
  
  Tensor xt({BATCH_SIZE, IMAGE_SIZE});
  Tensor yt({BATCH_SIZE, LABEL_SIZE});

  
  boost::timer::cpu_timer total;
  Adam opt;
  for(int i = 1; i <= 10; ++i) {
    boost::timer::cpu_timer timer;
    shuffle(trainImages, trainLabels, IMAGE_SIZE, LABEL_SIZE);
    float cost = 0;
    for(int j = 0; j < trainRows / BATCH_SIZE; j++) {
      size_t xBatch = IMAGE_SIZE * BATCH_SIZE;
      auto xbegin = trainImages.begin() + j * xBatch;
      auto xend = xbegin + xBatch;
      xt.set(xbegin, xend);
      
      size_t yBatch = LABEL_SIZE * BATCH_SIZE;
      auto ybegin = trainLabels.begin() + j * yBatch;
      auto yend = ybegin + yBatch;
      yt.set(ybegin, yend);
      
      g["x"] = xt;
      g["y"] = yt;
      
      opt(g, BATCH_SIZE);
      cost += g["cost"].val()[0];
    }
    std::cerr << "Epoch: " << i << " - Cost: " << cost / trainRows * BATCH_SIZE << " - " << timer.format(3, "%ws") << std::endl;
  }
  std::cerr << "Total: " << total.format(3, "%ws") << std::endl;

  std::vector<float> results;  
  for(int j = 0; j < testRows / BATCH_SIZE; j++) {
    size_t xBatch = IMAGE_SIZE * BATCH_SIZE;
    auto xbegin = testImages.begin() + j * xBatch;
    auto xend = xbegin + xBatch;
    xt.set(xbegin, xend);
    yt.set(0);  
      
    g["x"] = xt;
    g["y"] = yt;
    
    g.forward(BATCH_SIZE);
    
    std::vector<float> bResults;
    bResults << g["scores"].val();
    results.insert(results.end(), bResults.begin(), bResults.end());
  }
  
  size_t acc = 0;
  for (size_t i = 0; i < testLabels.size(); i += LABEL_SIZE) {
    size_t correct = 0;
    size_t proposed = 0;
    for (size_t j = 0; j < LABEL_SIZE; ++j) {
      if (testLabels[i + j])
        correct = j;
      if (results[i + j] > results[i + proposed])
        proposed = j;
    }
    acc += (correct == proposed);
  }
  std::cerr << "Accuracy: " << float(acc) / testRows << std::endl;
  
  return 0;
}
