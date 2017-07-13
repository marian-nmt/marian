#include <iostream>
#include <math.h>

#include "marian.h"
#include "layers/convolution.h"

using namespace marian;

bool test_vectors(const std::vector<float>& output, const std::vector<float>& corrent) {
  if (output.size() != corrent.size()) {
    return false;
  }

  for (size_t i = 0; i < output.size(); ++i) {
    if (fabsf(output[i] - corrent[i]) > 0.0001f) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  auto config = Config(argc, argv, ConfigMode::translating, false);
  auto graph = New<ExpressionGraph>(false);
  graph->setDevice(0);
  graph->reserveWorkspaceMB(128);

  int dimBatch = 2;
  int dimWord = 4;
  int batchLength = 5;
  int numLayers = 1;

  int elemNum = dimBatch * dimWord * batchLength * numLayers;

  std::vector<float> embData(elemNum);
  std::vector<float> embMask(elemNum);

  for (size_t i = 0; i < embData.size(); ++i) {
    embData[i] = 1.0f / (i + 1);
    if (i < dimBatch * batchLength) {
      embMask[i] = ((i < 8) ? 1 : 0);
    }
  }

  auto x = graph->param("x", {dimBatch, dimWord, batchLength},
                        keywords::init=inits::from_vector(embData));


  auto xMask = graph->constant({dimBatch, 1, batchLength},
                               keywords::init=inits::from_vector(embMask));

  auto convolution = Convolution("convolution", 3, 1, 3)(x, xMask);
  auto idx = graph->constant({elemNum, 1}, keywords::init=inits::zeros);
  auto ce = cross_entropy(convolution, idx);
  auto cost = mean(sum(ce, keywords::axis=2), keywords::axis=0);


  debug(x, "X");
  debug(cost, "COST");
  debug(convolution, "CONVOLUTION");

  graph->forward();
  graph->backward();
  std::vector<float> output(elemNum);
  std::vector<float> correct_forward = { 4.8, 5.4, 6, 6.6, 7.2, 7.8, 8.4, 9, 9.6, 10.4, 11.2, 12,
                                         12.8, 13.6, 14.4, 15.2, 16, 17, 18, 19, 20, 21, 22, 23,
                                         16, 16.8, 17.6, 18.4, 19.2, 20, 20.8, 21.6, 14.4, 15,
                                         15.6, 16.2, 16.8, 17.4, 18, 18.6};
  convolution->val() >> output;
  if (test_vectors(output, correct_forward)) {
    std::cerr << "OK" << std::endl;;
  } else {
    std::cerr << "DIFF" << std::endl;;
  }

  std::vector<float> grad(elemNum);
  std::vector<float> grad_in(elemNum);

  convolution->grad() >> grad_in;

  x->grad() >> grad;

  int ii = 0;
  for (auto v : grad_in) std::cerr << v << ((++ii % 4 != 0) ? " " : "\n");
  std::cerr << std::endl;

  ii = 0;
  for (auto v : grad) std::cerr << v << ((++ii % 4 != 0) ? " " : "\n");
  std::cerr << std::endl;




  return 0;
}
