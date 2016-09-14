
#include "marian.h"
#include "mnist.h"
#include "npz_converter.h"

using namespace marian;
using namespace keywords;

int main(int argc, char** argv) {
  const size_t IMAGE_SIZE = 784;
  const size_t LABEL_SIZE = 10;
  int numofdata;

  std::cerr << "Loading test set...";
  std::vector<float> testImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  std::vector<float>testLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numofdata, LABEL_SIZE);
  std::cerr << "\tDone." << std::endl;

  std::cerr << "Loading model params...";
  NpzConverter converter("../scripts/test_model/model.npz");

  std::vector<float> wData;
  Shape wShape;
  converter.Load("weights", wData, wShape);

  std::vector<float> bData;
  Shape bShape;
  converter.Load("bias", bData, bShape);

  auto initW = [&wData](Tensor t) {
    thrust::copy(wData.begin(), wData.end(), t.begin());
  };

  auto initB = [&bData](Tensor t) {
    thrust::copy(bData.begin(), bData.end(), t.begin());
  };

  std::cerr << "\tDone." << std::endl;


  Expr x = input(shape={whatevs, IMAGE_SIZE}, name="X");

  Expr w = param(shape={IMAGE_SIZE, LABEL_SIZE}, name="W0", init=initW);
  Expr b = param(shape={1, LABEL_SIZE}, name="b0", init=initB);

  std::cerr << "Building model...";
  auto scores = dot(x, w) + b;
  auto predict = softmax(scores, axis=1, name="pred");
  std::cerr << "\tDone." << std::endl;

  Tensor xt({numofdata, IMAGE_SIZE});
  xt.Load(testImages);

  predict.forward(numofdata);

  auto results = predict.val();

  size_t acc = 0;

  for (size_t i = 0; i < testLabels.size(); i += LABEL_SIZE) {
    size_t correct = 0;
    size_t predicted = 0;
    for (size_t j = 0; j < LABEL_SIZE; ++j) {
      if (testLabels[i+j]) correct = j;
      if (results[i + j] > results[i + predicted]) predicted = j;
    }
    acc += (correct == predicted);
    std::cerr << "corect: " << correct << " | " << predicted <<  "(";
    for (size_t j = 0; j < LABEL_SIZE; ++j) {
      std::cerr << results[i+j] << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << "ACC: " << float(acc)/numofdata << std::endl;

  return 0;
}
