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

void construct(ExpressionGraphPtr g,
               bool fast) {
  g->clear();

  int dimBatch = 1;

  int dimSrcVoc = g->get("Wemb") ? g->get("Wemb")->shape()[0] : 85000;
  int dimSrcEmb = g->get("Wemb") ? g->get("Wemb")->shape()[1] : 500;
  int dimEncState = g->get("encoder_U") ? g->get("encoder_U")->shape()[0] : 1024;

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

  auto buildEncoderGRU3 = [=](const std::string& prefix){
    auto Uz = g->param(prefix + "_Uz", {dimEncState, dimEncState},
                            init=glorot_uniform);
    auto Ur = g->param(prefix + "_Ur", {dimEncState, dimEncState},
                            init=glorot_uniform);

    auto Wz = g->param(prefix + "_Wz", {dimSrcEmb, dimEncState},
                            init=glorot_uniform);
    auto Wr = g->param(prefix + "_Wr", {dimSrcEmb, dimEncState},
                            init=glorot_uniform);

    auto bz = g->param(prefix + "_bz", {1, dimEncState}, init=zeros);
    auto br = g->param(prefix + "_br", {1, dimEncState}, init=zeros);

    auto Ux = g->param(prefix + "_Ux", {dimEncState, dimEncState},
                      init=glorot_uniform);

    auto Wx = g->param(prefix + "_Wx", {dimSrcEmb, dimEncState},
                       init=glorot_uniform);

    auto bx = g->param(prefix + "_bx", {1, dimEncState}, init=zeros);

    ParametersGRUFast encParams;
    encParams.U = concatenate({Ur, Uz, Ux}, 1);
    encParams.W = concatenate({Wr, Wz, Wx}, 1);
    encParams.b = concatenate({br, bz, bx}, 1);

    return RNN<GRUFast>(encParams);
  };

  auto x = name(g->ones(shape={dimBatch, dimSrcEmb}), "x");
  auto encStartState = name(g->ones(shape={dimBatch, dimEncState}), "start");

  if(fast) {
    auto encForward = buildEncoderGRU3("encoder");
    name(encForward.apply({x}, encStartState).back(), "out");
  }
  else {
    auto encForward = buildEncoderGRU("encoder");
    name(encForward.apply({x}, encStartState).back(), "out");
  }
}

int main(int argc, char** argv) {

  auto g1 = New<ExpressionGraph>();
  auto g2 = New<ExpressionGraph>();

  construct(g1, false);

  g1->forward();
  std::cerr << g1->get("out")->val()->debug() << std::endl;
  g1->backward();

  construct(g2, true);

  g2->forward();
  std::cerr << g2->get("out")->val()->debug() << std::endl;
  g2->backward();

  std::cerr << "s" << std::endl;
  std::cerr << g1->get("start")->grad()->debug() << std::endl;
  std::cerr << g2->get("start")->grad()->debug() << std::endl;

  for(std::string name : {"Ur", "Uz", "Ux",
                          "Wr", "Wz", "Wx",
                          "br", "bz", "bx"}) {
    std::cerr << name << std::endl;
    std::cerr << g1->get("encoder_" + name)->grad()->debug() << std::endl;
    std::cerr << g2->get("encoder_" + name)->grad()->debug() << std::endl;
  }


  return 0;
}
