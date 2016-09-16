
#include "marian.h"
#include "mnist.h"
#include "sgd.h"

using namespace std;

int main(int argc, char** argv) {
  const size_t IMAGE_SIZE = 784;
  const size_t LABEL_SIZE = 10;
  int numofdata;

  vector<float> trainImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  vector<float>trainLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numofdata, LABEL_SIZE);

  using namespace marian;
  using namespace keywords;

  ExpressionGraph g;
  
  Expr x = named(g.input(shape={whatevs, IMAGE_SIZE}), "x");
  Expr y = named(g.input(shape={whatevs, LABEL_SIZE}), "y");

  Expr w = named(g.param(shape={IMAGE_SIZE, LABEL_SIZE}), "w");
  Expr b = named(g.param(shape={1, LABEL_SIZE}), "b");

  std::vector<Expr*> params;
  params.push_back(&w);
  params.push_back(&b);

  auto scores = dot(x, w) + b;
  auto lr = softmax_fast(scores);
  auto cost = named(-mean(sum(y * log(lr), axis=1), axis=0), "cost");
  cerr << "lr=" << lr.Debug() << endl;

  SGD opt(g, 0.9, trainImages, IMAGE_SIZE, trainLabels, LABEL_SIZE, 3, 24);
  opt.Run();
  return 0;
}
