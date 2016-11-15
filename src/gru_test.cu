#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "rnn.h"
#include "batch_generator.h"

using namespace marian;
using namespace keywords;
using namespace data;

typedef DeviceVector<size_t> WordBatch;
typedef std::vector<WordBatch> SentBatch;

void construct(ExpressionGraphPtr g,
               const SentBatch& srcSentenceBatch) {
  g->clear();

  int dimSrcVoc = 30000;
  int dimSrcEmb = 512;
  int dimEncState = 1024;
  int dimBatch = 1;

  auto Wemb = g->param("Wemb", {dimSrcVoc, dimSrcEmb}, init=uniform());

  std::vector<Expr> inputs;
  for(auto& srcWordBatch : srcSentenceBatch) {
    auto x = rows(Wemb, srcWordBatch);
    inputs.push_back(x);
    dimBatch = srcWordBatch.size();
  }

  auto encoder = [=](const std::string& prefix){
    ParametersGRU encParams;
    encParams.Uz = g->param(prefix + "_Uz", {dimEncState, dimEncState},
                            init=uniform());
    encParams.Ur = g->param(prefix + "_Ur", {dimEncState, dimEncState},
                            init=uniform());

    encParams.Wz = g->param(prefix + "_Wz", {dimSrcEmb, dimEncState},
                            init=uniform());
    encParams.Wr = g->param(prefix + "_Wr", {dimSrcEmb, dimEncState},
                            init=uniform());

    encParams.bz = g->param(prefix + "_bz", {1, dimEncState}, init=zeros);
    encParams.br = g->param(prefix + "_br", {1, dimEncState}, init=zeros);

    encParams.Ux = g->param(prefix + "_Ux", {dimEncState, dimEncState},
                            init=uniform());
    encParams.Wx = g->param(prefix + "_Wx", {dimSrcEmb, dimEncState},
                            init=uniform());
    encParams.bx = g->param(prefix + "_bx", {1, dimEncState}, init=zeros);

    return RNN<GRU>(encParams);
  };

  auto encStartState = g->zeros(shape={dimBatch, dimEncState});

  auto encForward = encoder("encoder");
  auto statesForward = encForward.apply(inputs.begin(), inputs.end(),
                                        encStartState);

  /*
  auto encBackward = encoder("encoder_r");
  auto statesBackward = encBackward.apply(inputs.rbegin(), inputs.rend(),
                                          encStartState);

  std::vector<Expr> joinedStates;
  for(auto itFw = statesForward.begin(), auto itBw = statesBackward.rbegin();
      itFw != statesForward.end(); itFw++, itBw++)
    joinedStates.push_back(concatenate({*itFw, *itBw}, axis=1));

  auto encoder = concatenate(joinedStates, axis=2)
  auto decStartState = mean(encoder, axis=2);
  */
}

SentBatch generateBatch(size_t batchSize) {
  size_t length = rand() % 40 + 10;
  return SentBatch(length, WordBatch(batchSize));
}

int main(int argc, char** argv) {
  auto g = New<ExpressionGraph>();

  size_t batchSize = 80;

  boost::timer::cpu_timer timer;
  for(int i = 1; i <= 1000; ++i) {
    g->clear();

    // fake batch
    auto batch = generateBatch(batchSize);
    construct(g, batch);

    g->forward();
    if(i % 100 == 0)
      std::cout << i << std::endl;
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
