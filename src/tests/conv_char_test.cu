#include <iostream>
#include <math.h>

#include "marian.h"
#include "layers/highway.h"
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
  auto config = Config(argc, argv, false);
  auto graph = New<ExpressionGraph>(false);
  graph->setDevice(3);
  graph->reserveWorkspaceMB(128);

  int dimBatch = 2;
  int dimWord = 4;
  int batchLength = 3;
  int numLayers = 1;

  int elemNum = dimBatch * dimWord * batchLength * numLayers;

  std::vector<float> embData(elemNum);
  std::vector<float> embMask(dimBatch * batchLength);

  for (size_t i = 0; i < embData.size(); ++i) {
    embData[i] = float(1) / (i + 1.0f);
  }

  for (auto& v : embMask) {
    v = 1.0f;
  }

  auto x = graph->param("x", {dimBatch, dimWord, batchLength},
                        keywords::init=inits::from_vector(embData));


  auto xMask = graph->constant({dimBatch, 1, batchLength},
                               keywords::init=inits::from_vector(embMask));

  std::vector<Expr> convs({Convolution("convolution_1", 1, dimWord, 200)(x, xMask),
                           Convolution("convolution_2", 2, dimWord, 200)(x, xMask),
                           Convolution("convolution_3", 3, dimWord, 250)(x, xMask),
                           Convolution("convolution_4", 4, dimWord, 250)(x, xMask),
                           Convolution("convolution_5", 5, dimWord, 300)(x, xMask),
                           Convolution("convolution_6", 6, dimWord, 300)(x, xMask),
                           Convolution("convolution_7", 7, dimWord, 300)(x, xMask),
                           Convolution("convolution_8", 8, dimWord, 300)(x, xMask)});


  auto convolution = concatenate(convs, keywords::axis=1);


  auto r = relu(convolution);
  // debug(r, "R");
  auto maxPooling = MaxPooling("max_pooling", 5, 1, 5, 1)(r);
  auto highway = Highway("highway", 4)(maxPooling);
  auto idx = graph->constant({4, 1}, keywords::init=inits::zeros);
  auto ce = cross_entropy(highway, idx);
  auto cost = mean(sum(ce, keywords::axis=2), keywords::axis=0);


  debug(x, "x");
  debug(cost, "COST");
  debug(convolution, "CONVOLUTION");
  debug(r, "RELU");
  debug(maxPooling, "MAXP LAST");
  debug(highway, "highway");
  debug(ce, "ce");
  debug(convs.front(), "CONV 1");

  graph->forward();
  graph->backward();
  std::vector<float> output;
  std::vector<float> output2;
  // std::vector<float> correct_forward = { 4.8, 5.4, 6, 6.6, 7.2, 7.8, 8.4, 9, 9.6, 10.4, 11.2, 12,
                                         // 12.8, 13.6, 14.4, 15.2, 16, 17, 18, 19, 20, 21, 22, 23,
                                         // 16, 16.8, 17.6, 18.4, 19.2, 20, 20.8, 21.6, 14.4, 15,
                                         // 15.6, 16.2, 16.8, 17.4, 18, 18.6};
  // convolution->val() >> output;
  // maxPooling->val() >> output2;

  // if (test_vectors(output, correct_forward)) {
    // std::cerr << "OK" << std::endl;;
  // } else {
    // std::cerr << "DIFF" << std::endl;;
  // }

  // std::vector<float> grad(elemNum);
  // std::vector<float> grad_in(elemNum);

  // convolution->grad() >> grad_in;

  // x->grad() >> grad;

  // int ii = 0;
  // for (auto v : output) std::cerr << v << " ";
  // std::cerr << std::endl;

  for (auto v : output2) std::cerr << v << " ";
  std::cerr << std::endl;

  // ii = 0;
  // for (auto v : grad) std::cerr << v << ((++ii % 4 != 0) ? " " : "\n");
  // std::cerr << std::endl;




  return 0;
}
