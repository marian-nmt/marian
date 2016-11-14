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

void construct(ExpressionGraphPtr g, size_t length) {
  g->clear();

  int dim_i = 500;
  int dim_h = 1024;

  //ParametersGRU pGRU;
  //pGRU.Uz = g->param("Uz", {dim_h, dim_h}, init=uniform());
  //pGRU.Wz = g->param("Wz", {dim_i, dim_h}, init=uniform());
  ////pGRU.bz = nullptr; // g->param("bz", {1, dim_h}, init=zeros);
  //
  //pGRU.Ur = g->param("Ur", {dim_h, dim_h}, init=uniform());
  //pGRU.Wr = g->param("Wr", {dim_i, dim_h}, init=uniform());
  ////pGRU.br = nullptr; //g->param("br", {1, dim_h}, init=zeros);
  //
  //pGRU.Uh = g->param("Uh", {dim_h, dim_h}, init=uniform());
  //pGRU.Wh = g->param("Wh", {dim_i, dim_h}, init=uniform());
  ////pGRU.bh = nullptr; //g->param("bh", {1, dim_h}, init=zeros);

  ParametersTanh pTanh;
  pTanh.U = g->param("Uz", {dim_h, dim_h}, init=uniform());
  pTanh.W = g->param("Wz", {dim_i, dim_h}, init=uniform());
  pTanh.b = nullptr; // g->param("bz", {1, dim_h}, init=zeros);

  auto start = name(g->zeros(shape={whatevs, dim_h}), "s_0");
  std::vector<Expr> inputs;
  for(int i = 0; i < length; ++i) {
    auto x = name(g->input(shape={whatevs, dim_i}),
                  "x_" + std::to_string(i));
    inputs.push_back(x);
  }

  RNN<> rnn(pTanh);
  auto outputs = rnn.apply(inputs, start);

  //RNN<GRU> gru(pGRU);
  //auto outputs = gru.apply(inputs, start);
}

int main(int argc, char** argv) {
  auto g = New<ExpressionGraph>();

  boost::timer::cpu_timer timer;
  for(int i = 1; i <= 1000; ++i) {
    size_t length = rand() % 40 + 10; // random int from [10,50]
    g->clear();
    construct(g, length);

    BatchPtr batch(new Batch());
    for(int j = 0; j < length; ++j)
      batch->push_back(Input({80, 500}));

    g->forward(batch);
    if(i % 100 == 0)
      std::cout << i << std::endl;
  }
  std::cout << std::endl;
  std::cout << timer.format(5, "%ws") << std::endl;

  return 0;
}
