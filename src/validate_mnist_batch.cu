
#include "marian.h"
#include "mnist.h"
#include "npz_converter.h"

using namespace marian;
using namespace keywords;

int main(int argc, char** argv) {
  
  cudaSetDevice(0);
  
  const size_t IMAGE_SIZE = 784;
  const size_t LABEL_SIZE = 10;
  const size_t BATCH_SIZE = 24;
  int numofdata;

  std::cerr << "Loading test set...";
  std::vector<float> testImages = datasets::mnist::ReadImages("../examples/mnist/t10k-images-idx3-ubyte", numofdata, IMAGE_SIZE);
  std::vector<float> testLabels = datasets::mnist::ReadLabels("../examples/mnist/t10k-labels-idx1-ubyte", numofdata, LABEL_SIZE);
  std::cerr << "\tDone." << std::endl;

  std::cerr << "Loading model params...";
  NpzConverter converter("../scripts/test_model_multi/model.npz");

  std::vector<float> wData1;
  Shape wShape1;
  converter.Load("weights1", wData1, wShape1);
  
  std::vector<float> bData1;
  Shape bShape1;
  converter.Load("bias1", bData1, bShape1);
  
  std::vector<float> wData2;
  Shape wShape2;
  converter.Load("weights2", wData2, wShape2);
  
  std::vector<float> bData2;
  Shape bShape2;
  converter.Load("bias2", bData2, bShape2);

  auto initW1 = [wData1](Tensor t) {
    t.set(wData1);
  };

  auto initB1 = [bData1](Tensor t) {
    t.set(bData1);
  };
  
  auto initW2 = [wData2](Tensor t) {
    t.set(wData2);
  };

  auto initB2 = [bData2](Tensor t) {
    t.set(bData2);
  };

  std::cerr << "\tDone." << std::endl;


  auto x = input(shape={whatevs, IMAGE_SIZE}, name="X");
  auto y = input(shape={whatevs, LABEL_SIZE}, name="Y");

  auto w1 = param(shape={IMAGE_SIZE, 100}, name="W0", init=initW1);
  auto b1 = param(shape={1, 100}, name="b0", init=initB1);
  auto w2 = param(shape={100, LABEL_SIZE}, name="W1", init=initW2);
  auto b2 = param(shape={1, LABEL_SIZE}, name="b1", init=initB2);

  std::cerr << "Building model...";
  auto layer1 = tanh(dot(x, w1) + b1);
  auto layer2 = softmax(dot(layer1, w2) + b2, axis=1, name="layer2");
  auto predict = layer2;

  std::cerr << "Done." << std::endl;

  Tensor xt({BATCH_SIZE, IMAGE_SIZE});

  size_t acc = 0;
  size_t startId = 0;
  size_t endId = startId + BATCH_SIZE;

  while (endId < numofdata) {
    std::vector<float> tmp(testImages.begin() + (startId * IMAGE_SIZE),
                           testImages.begin() + (endId * IMAGE_SIZE));
    xt << tmp;
    x = xt;

    predict.forward(BATCH_SIZE);

    std::vector<float> results(LABEL_SIZE * BATCH_SIZE);
    results << predict.val();

    for (size_t i = 0; i < BATCH_SIZE * LABEL_SIZE; i += LABEL_SIZE) {
      size_t correct = 0;
      size_t predicted = 0;
      for (size_t j = 0; j < LABEL_SIZE; ++j) {
        if (testLabels[startId * LABEL_SIZE + i + j]) correct = j;
        if (results[i + j] > results[i + predicted]) predicted = j;
      }
      /*std::cerr << "CORRECT: " << correct << " PREDICTED: " << predicted << std::endl;*/
      acc += (correct == predicted);
    }

    startId += BATCH_SIZE;
    endId += BATCH_SIZE;
  }
  if (endId != numofdata) {
    endId = numofdata;
    if (endId - startId > 0) {
      std::vector<float> tmp(testImages.begin() + (startId * IMAGE_SIZE),
                             testImages.begin() + (endId * IMAGE_SIZE));
      xt << tmp;
      x = xt;

      predict.forward(endId - startId);

      std::vector<float> results(LABEL_SIZE * BATCH_SIZE);
      results << predict.val();

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
