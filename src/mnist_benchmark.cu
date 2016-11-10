#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "trainer.h"
#include "models/feedforward.h"

using namespace marian;
using namespace keywords;
using namespace data;
using namespace models;

int main(int argc, char** argv) {

  auto g = New<ExpressionGraph>();
  auto x = name(g->input(shape={whatevs, 784}),
                "x");
  auto y = name(g->input(shape={whatevs, 10}),
                "y");

  auto w = name(g->param(shape={784, 10}, init=uniform()), "W");
  auto b = name(g->param(shape={1, 10}, init=zeros), "b");


  auto cost = name(mean(sum(log(softmax(dot(x, w) + b)) * y), axis=0),
                   "cost");


  g->graphviz("mnist_logistic.dot");




  return 0;
}
