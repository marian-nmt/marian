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

using namespace marian;
using namespace keywords;
using namespace data;

typedef DeviceVector<size_t> WordBatch;
typedef std::vector<WordBatch> SentBatch;

void construct(ExpressionGraphPtr g,
               const SentBatch& srcSentenceBatch) {
  g->clear();

  int dimSrcVoc = 80000;
  int dimSrcEmb = 512;
  int dimEncState = 1024;
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
    ParametersGRU encParams;
    encParams.Uz = g->param(prefix + "_Uz", {dimEncState, dimEncState},
                            init=glorot_uniform);
    encParams.Ur = g->param(prefix + "_Ur", {dimEncState, dimEncState},
                            init=glorot_uniform);
  
    encParams.Wz = g->param(prefix + "_Wz", {dimSrcEmb, dimEncState},
                            init=glorot_uniform);
    encParams.Wr = g->param(prefix + "_Wr", {dimSrcEmb, dimEncState},
                            init=glorot_uniform);
  
    encParams.bz = g->param(prefix + "_bz", {1, dimEncState}, init=zeros);
    encParams.br = g->param(prefix + "_br", {1, dimEncState}, init=zeros);
  
    encParams.Ux = g->param(prefix + "_Ux", {dimEncState, dimEncState},
                            init=glorot_uniform);
    encParams.Wx = g->param(prefix + "_Wx", {dimSrcEmb, dimEncState},
                            init=glorot_uniform);
    encParams.bx = g->param(prefix + "_bx", {1, dimEncState}, init=zeros);
  
    return RNN<GRU>(encParams);
  };

  auto buildEncoderGRU2 = [=](const std::string& prefix){
    ParametersGRUFast encParams;
    encParams.U = g->param(prefix + "_U", {dimEncState, 3 * dimEncState},
                           init=glorot_uniform);
  
    encParams.W = g->param(prefix + "_W", {dimSrcEmb, 3 * dimEncState},
                           init=glorot_uniform);
  
    encParams.b = g->param(prefix + "_b", {1, 3 * dimEncState}, init=zeros);
  
    return RNN<GRUFast>(encParams);
  };

  auto buildEncoderGRU3 = [=](const std::string& prefix){
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
    encParams.U = transpose(concatenate({transpose(U), transpose(Ux)}));
    encParams.W = transpose(concatenate({transpose(W), transpose(Wx)}));
    encParams.b = transpose(concatenate({transpose(b), transpose(bx)}));

    return RNN<GRUFast>(encParams);
  };

  auto encStartState = name(g->zeros(shape={dimBatch, dimEncState}), "start");

  auto encForward = buildEncoderGRU3("encoder");
  auto statesForward = encForward.apply(inputs.begin(), inputs.end(),
                                        encStartState);

  auto encBackward = buildEncoderGRU3("encoder_r");
  auto statesBackward = encBackward.apply(inputs.rbegin(), inputs.rend(),
                                          encStartState);

  std::vector<Expr> joinedStates;
  auto itFw = statesForward.begin();
  auto itBw = statesBackward.rbegin();
  while(itFw != statesForward.end()) {
    // add proper axes
    joinedStates.push_back(concatenate({*itFw++, *itBw++}));
  }

  // add proper axes and make this a 3D tensor
  auto encContext = name(concatenate(joinedStates), "context");

  //auto decStartState = mean(encContext);
}

SentBatch generateBatch(size_t batchSize) {
  size_t length = rand() % 40 + 10;
  return SentBatch(length, WordBatch(batchSize));
}

int main(int argc, char** argv) {
  cudaSetDevice(2);

  auto g = New<ExpressionGraph>();

  size_t batchSize = 80;

  boost::timer::cpu_timer timer;
  for(int i = 1; i <= 1000; ++i) {
    g->clear();

    // fake batch
    auto batch = generateBatch(batchSize);
    construct(g, batch);
    //g->graphviz("nematus.dot");
    //exit(1);

    g->forward();
    //g->backward();
    if(i % 100 == 0)
      std::cout << i << std::endl;
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
