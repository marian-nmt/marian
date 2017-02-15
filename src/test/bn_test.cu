#include <iostream>
#include <cuda.h>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>

#include "marian.h"

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;
  using namespace keywords;

  auto options = New<Config>(argc, argv, false);

  std::vector<float> temp(128 * 512);
  std::vector<float> indeces(128, 0.f);

  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine(rnd_device());
  mersenne_engine.seed(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  auto gen = std::bind(dist, mersenne_engine);
  std::generate(std::begin(temp), std::end(temp), gen);

  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    auto x = graph->param("x", {128, 512}, init=inits::from_vector(temp));
    auto gamma = graph->param("gamma", {1, 512}, init=inits::from_value(1.0));
    auto beta = graph->param("beta", {1, 512}, init=inits::zeros);

    auto mju = mean(x, keywords::axis=1);
    auto xmmju = x - mju;
    auto std = sqrt(mean(square(xmmju), keywords::axis=1), 1e-9);
    auto y = gamma * (xmmju / std) + beta;

    auto idx = graph->constant(shape={(int)indeces.size(), 1},
                               init=inits::from_vector(indeces));
    auto ce = cross_entropy(y, idx);
    auto cost = mean(sum(ce, keywords::axis=2), keywords::axis=0);

    debug(x, "x");
    debug(y, "y");
    debug(cost, "cost");

    graph->forward();
    graph->backward();
  }

  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    auto x = graph->param("x", {128, 512}, init=inits::from_vector(temp));
    auto gamma = graph->param("gamma", {1, 512}, init=inits::from_value(1.0));
    auto beta = graph->param("beta", {1, 512}, init=inits::zeros);

    auto y = layer_norm(x, gamma, beta);

    auto idx = graph->constant(shape={(int)indeces.size(), 1},
                               init=inits::from_vector(indeces));
    auto ce = cross_entropy(y, idx);
    auto cost = mean(sum(ce, keywords::axis=2), keywords::axis=0);

    debug(x, "x");
    debug(y, "y");
    debug(cost, "cost");

    graph->forward();
    graph->backward();
  }

  return 0;
}
