
#include "marian.h"
#include "mnist.h"
#include "npz_converter.h"

using namespace marian;
using namespace keywords;

int main(int argc, char** argv) {
  
  cudaSetDevice(0);
  
  const size_t IMAGE_SIZE = 784;
  const size_t LABEL_SIZE = 10;
  
  std::cerr << "Loading test set...";
  std::vector<float> testImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  std::vector<float> testLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numofdata, LABEL_SIZE);
  std::cerr << "Done." << std::endl;

  const size_t BATCH_SIZE = testLabels.size();
  
  std::cerr << "Loading model params...";
  NpzConverter converter("../scripts/test_model/model.npz");

  std::vector<float> wData, bData;
  Shape wShape, bShape;
  converter.Load("weights", wData, wShape);
  converter.Load("bias", bData, bShape);

  std::cerr << "Done." << std::endl;

  auto x = input(shape={whatevs, IMAGE_SIZE}, name="X");
  auto y = input(shape={whatevs, LABEL_SIZE}, name="Y");
  
  auto w = param(shape={IMAGE_SIZE, LABEL_SIZE}, name="W0",
                 init=[wData](Tensor t) { t.set(wData); });
  auto b = param(shape={1, LABEL_SIZE}, name="b0",
                 init=[bData](Tensor t) {t.set(bData); });

  std::cerr << "Building model...";
  auto predict = softmax(dot(x, w) + b,
                         axis=1, name="pred");
  auto graph = -mean(sum(y * log(predict), axis=1),
                     axis=0, name="cost");
  
  std::cerr << "Done." << std::endl;

  Tensor xt({numofdata, IMAGE_SIZE});
  Tensor yt({numofdata, LABEL_SIZE});
  
  x = xt << testImages;
  y = yt << testLabels;
  
  graph.forward(BATCH_SIZE);
  auto results = predict.val();
  graph.backward();
  
  std::vector<float> resultsv(results.size());
  resultsv << results;
  
  std::cerr << b.grad().Debug() << std::endl;

  size_t acc = 0;
  for (size_t i = 0; i < testLabels.size(); i += LABEL_SIZE) {
    size_t correct = 0;
    size_t predicted = 0;
    for (size_t j = 0; j < LABEL_SIZE; ++j) {
      if (testLabels[i+j]) correct = j;
      if (resultsv[i + j] > resultsv[i + predicted]) predicted = j;
    }
    acc += (correct == predicted);
    //std::cerr << correct << " | " << predicted <<  " ( ";
    //for (size_t j = 0; j < LABEL_SIZE; ++j) {
    //  std::cerr << resultsv[i+j] << " ";
    //}
    //std::cerr << ")" << std::endl;
  }
  std::cerr << "Accuracy: " << float(acc)/BATCH_SIZE << std::endl;

  return 0;
}
