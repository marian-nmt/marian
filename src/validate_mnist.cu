
#include "marian.h"
#include "mnist.h"
#include "npz_converter.h"

using namespace marian;
using namespace keywords;

int main(int argc, char** argv) {
  const size_t IMAGE_SIZE = 784;
  const size_t LABEL_SIZE = 10;
  const size_t BATCH_SIZE = 24;
  int numofdata;

  std::cerr << "Loading test set...";
  std::vector<float> testImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  std::vector<float> testLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numofdata, LABEL_SIZE);
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

  Tensor xt({BATCH_SIZE, IMAGE_SIZE});
  /* xt.Load(testImages); */
  /* x = xt; */

  size_t acc = 0;
  size_t startId = 0;
  size_t endId = startId + BATCH_SIZE;

  while (endId < numofdata) {
    std::vector<float> tmp(testImages.begin() + (startId * IMAGE_SIZE),
                           testImages.begin() + (endId * IMAGE_SIZE));
    xt.Load(tmp);
    x = xt;

    predict.forward(BATCH_SIZE);

    thrust::host_vector<float> results(predict.val().begin(), predict.val().begin() + LABEL_SIZE * BATCH_SIZE);

    for (size_t i = 0; i < BATCH_SIZE * LABEL_SIZE; i += LABEL_SIZE) {
      size_t correct = 0;
      size_t predicted = 0;
      for (size_t j = 0; j < LABEL_SIZE; ++j) {
        if (testLabels[startId * LABEL_SIZE + i + j]) correct = j;
        if (results[i + j] > results[i + predicted]) predicted = j;
      }
      acc += (correct == predicted);
    }

    startId += BATCH_SIZE;
    endId += BATCH_SIZE;
  }
  if (endId != numofdata) {
    endId = numofdata;
    if (endId - startId >= 0) {
      std::vector<float> tmp(testImages.begin() + (startId * IMAGE_SIZE),
                             testImages.begin() + (endId * IMAGE_SIZE));
      xt.Load(tmp);
      x = xt;

      predict.forward(endId - startId);

      thrust::host_vector<float> results(predict.val().begin(), predict.val().begin() + LABEL_SIZE * (endId - startId));

      for (size_t i = 0; i < (endId - startId) * LABEL_SIZE; i += LABEL_SIZE) {
        size_t correct = 0;
        size_t predicted = 0;
        for (size_t j = 0; j < LABEL_SIZE; ++j) {
          if (testLabels[startId * LABEL_SIZE + i + j]) correct = j;
          if (results[i + j] > results[i + predicted]) predicted = j;
        }
        acc += (correct == predicted);
      }
    }
  }

  std::cerr << "ACC: " << float(acc)/numofdata << std::endl;

  return 0;
}
