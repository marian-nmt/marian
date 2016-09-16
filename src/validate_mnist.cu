
#include "marian.h"
#include "mnist.h"
#include "npz_converter.h"
#include "sgd.h"

using namespace marian;
using namespace keywords;

const size_t IMAGE_SIZE = 784;
const size_t LABEL_SIZE = 10;
int BATCH_SIZE = 10000;

ExpressionGraph build_graph() {
  std::cerr << "Loading model params...";
  NpzConverter converter("../scripts/test_model_single/model.npz");

  std::vector<float> wData, bData;
  Shape wShape, bShape;
  converter.Load("weights", wData, wShape);
  converter.Load("bias", bData, bShape);
  std::cerr << "Done." << std::endl;

  std::cerr << "Building model...";
  
  ExpressionGraph g;
  auto x = named(g.input(shape={whatevs, IMAGE_SIZE}), "x");
  auto y = named(g.input(shape={whatevs, LABEL_SIZE}), "y");
  
  auto w = named(g.param(shape={IMAGE_SIZE, LABEL_SIZE},
                         init=from_vector(wData)), "w");
  auto b = named(g.param(shape={1, LABEL_SIZE},
                         init=from_vector(bData)), "b");
  
                         
  auto probs = named(
    softmax(dot(x, w) + b),
    "probs"
  );
  
  auto cost = named(
    -mean(sum(y * log(probs), axis=1), axis=0),
    "cost"
  );
  
  std::cerr << "Done." << std::endl;
  return g;
}

int main(int argc, char** argv) {

  std::cerr << "Loading test set...";
  std::vector<float> testImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", BATCH_SIZE, IMAGE_SIZE);
  std::vector<float> testLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", BATCH_SIZE, LABEL_SIZE);
  std::cerr << "Done." << std::endl;

  ExpressionGraph g = build_graph();
  
  Tensor xt({BATCH_SIZE, IMAGE_SIZE});
  Tensor yt({BATCH_SIZE, LABEL_SIZE});
  
  g["x"] = (xt << testImages);
  g["y"] = (yt << testLabels);
  
  Adam opt;
  for(size_t j = 0; j < 20; ++j) {
    for(size_t i = 0; i < 60; ++i) {
      opt(g, BATCH_SIZE);
    }
    std::cerr << g["cost"].val()[0] << std::endl;
  }
  
  //std::cout << g.graphviz() << std::endl;
  
  std::vector<float> results;
  results << g["probs"].val();
  
  size_t acc = 0;
  for (size_t i = 0; i < testLabels.size(); i += LABEL_SIZE) {
    size_t correct = 0;
    size_t proposed = 0;
    for (size_t j = 0; j < LABEL_SIZE; ++j) {
      if (testLabels[i+j]) correct = j;
      if (results[i + j] > results[i + proposed]) proposed = j;
    }
    acc += (correct == proposed);
  }
  std::cerr << "Cost: " << g["cost"].val()[0] <<  " - Accuracy: " << float(acc) / BATCH_SIZE << std::endl;
  return 0;
}
