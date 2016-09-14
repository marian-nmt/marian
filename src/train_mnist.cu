
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

  Expr x = input(shape={whatevs, IMAGE_SIZE}, name="X");
  Expr y = input(shape={whatevs, LABEL_SIZE}, name="Y");

  Expr w = param(shape={IMAGE_SIZE, LABEL_SIZE}, name="W0");
  Expr b = param(shape={1, LABEL_SIZE}, name="b0");

  std::vector<Expr*> params;
  params.push_back(&w);
  params.push_back(&b);

  auto scores = dot(x, w) + b;
  auto lr = softmax_fast(scores, axis=1, name="pred");
  auto cost = -mean(sum(y * log(lr), axis=1), axis=0, name="cost");
  cerr << "lr=" << lr.Debug() << endl;

  SGD opt(cost, x, y, params, 0.9, trainImages, IMAGE_SIZE, trainLabels, LABEL_SIZE, 3, 24);
  opt.Run();
  return 0;
}
