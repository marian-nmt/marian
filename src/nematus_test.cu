#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "rnn.h"
#include "batch_generator.h"
#include "param_initializers.h"

#include "cnpy/cnpy.h"

using namespace marian;
using namespace keywords;
using namespace data;

typedef DeviceVector<size_t> WordBatch;
typedef std::vector<WordBatch> SentBatch;

void load(ExpressionGraphPtr g, const std::string& name) {
  auto model = cnpy::npz_load(name);

  auto chosen = {
    "Wemb",

    "encoder_U", "encoder_W", "encoder_b",
    "encoder_Ux", "encoder_Wx", "encoder_bx",

    "encoder_r_U", "encoder_r_W", "encoder_r_b",
    "encoder_r_Ux", "encoder_r_Wx", "encoder_r_bx"
  };

  for(auto name : chosen) {
    Shape shape;

    if(model[name].shape.size() == 2) {
      shape[0] = model[name].shape[0];
      shape[1] = model[name].shape[1];
    }
    else if(model[name].shape.size() == 1) {
      shape[0] = 1;
      shape[1] = model[name].shape[0];
    }

    auto p = g->param(name, shape, init=from_numpy(model[name]));
  }
}

void construct(ExpressionGraphPtr g,
               const SentBatch& srcSentenceBatch) {
  g->clear();

  int dimSrcVoc = g->get("Wemb") ? g->get("Wemb")->shape()[0] : 85000;
  int dimSrcEmb = g->get("Wemb") ? g->get("Wemb")->shape()[1] : 500;
  int dimEncState = g->get("encoder_U") ? g->get("encoder_U")->shape()[0] : 1024;

  int dimBatch = 1;

  auto Wemb = g->param("Wemb", {dimSrcVoc, dimSrcEmb}, init=glorot_uniform);

  std::vector<Expr> inputs;
  size_t i = 0;
  for(auto& srcWordBatch : srcSentenceBatch) {
    auto x = name(rows(Wemb, srcWordBatch), "x_" + std::to_string(i++));
    inputs.push_back(x);
    dimBatch = srcWordBatch.size();
  }

  auto buildEncoderGRU = [=](const std::string& prefix){
    auto U = g->param(prefix + "_U", {dimEncState, 2 * dimEncState},
                      init=glorot_uniform);

    auto W = g->param(prefix + "_W", {dimSrcEmb, 2 * dimEncState},
                      init=glorot_uniform);

    auto b = g->param(prefix + "_b", {1, 2 * dimEncState}, init=zeros);

    auto Ux = g->param(prefix + "_Ux", {dimEncState, dimEncState},
                      init=glorot_uniform);

    auto Wx = g->param(prefix + "_Wx", {dimSrcEmb, dimEncState},
                       init=glorot_uniform);

    auto bx = g->param(prefix + "_bx", {1, dimEncState}, init=zeros);

    ParametersGRUFast encParams;
    encParams.U = concatenate({U, Ux}, 1);
    encParams.W = concatenate({W, Wx}, 1);
    encParams.b = concatenate({b, bx}, 1);

    return RNN<GRUFast>(encParams);
  };

  auto encStartState = name(g->zeros(shape={dimBatch, dimEncState}), "start");

  auto encForward = buildEncoderGRU("encoder");
  auto statesForward = encForward.apply(inputs.begin(), inputs.end(),
                                        encStartState);

  auto encBackward = buildEncoderGRU("encoder_r");
  auto statesBackward = encBackward.apply(inputs.rbegin(), inputs.rend(),
                                          encStartState);

  std::vector<Expr> joinedStates;
  auto itFw = statesForward.begin();
  auto itBw = statesBackward.rbegin();
  while(itFw != statesForward.end()) {
    // add proper axes
    joinedStates.push_back(concatenate({*itFw++, *itBw++}, 1));
  }

  // add proper axes and make this a 3D tensor
  auto encContext = name(concatenate(joinedStates, 2), "context");

  //auto decStartState = mean(encContext, axis=2);
}

SentBatch generateBatch(size_t batchSize) {
  size_t length = rand() % 40 + 10;
  return SentBatch(length, WordBatch(batchSize));

  // das ist ein kleiner test . </s>
  //return SentBatch({
  //  WordBatch(batchSize, 13),
  //  WordBatch(batchSize, 15),
  //  WordBatch(batchSize, 20),
  //  WordBatch(batchSize, 8306),
  //  WordBatch(batchSize, 4),
  //  WordBatch(batchSize, 0)
  //});
}

int main(int argc, char** argv) {
  cudaSetDevice(0);

  auto g = New<ExpressionGraph>();
  load(g, "/home/marcinj/Badania/amunmt/test2/model.npz");

  size_t batchSize = 80;

  boost::timer::cpu_timer timer;
  for(int i = 1; i <= 1000; ++i) {
    g->clear();

    // fake batch
    auto batch = generateBatch(batchSize);
    construct(g, batch);

    g->forward();
    //exit(0);
    g->backward();
    if(i % 100 == 0)
      std::cout << i << std::endl;
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
