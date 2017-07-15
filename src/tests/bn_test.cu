#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "layers/generic.h"
#include "marian.h"

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;
  using namespace keywords;

  auto options = New<Config>(argc, argv, ConfigMode::training, false);

  int batchSize = 128;

  std::vector<float> temp(batchSize * 3072);
  std::vector<float> temp2(3072 * 3072);
  std::vector<float> indeces(batchSize, 0.f);

  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine(rnd_device());
  mersenne_engine.seed(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  auto gen = std::bind(dist, mersenne_engine);
  std::generate(std::begin(temp), std::end(temp), gen);
  std::generate(std::begin(temp2), std::end(temp2), gen);

  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    auto x
        = graph->param("x", {batchSize, 3072}, init = inits::from_vector(temp));
    auto gamma
        = graph->param("gamma", {1, 3072}, init = inits::from_value(2.0));
    auto beta = graph->param("beta", {1, 3072}, init = inits::zeros);

    auto y = layer_norm(x, gamma, beta);

    auto yLogitsL1 = Dense(
        "ff_logit_l1", 512, activation = act::tanh, normalize = true)(y, y, y);

    auto yLogitsL2 = Dense("ff_logit_l2", 50000)(yLogitsL1);

    auto idx = graph->constant({(int)indeces.size(), 1},
                               init = inits::from_vector(indeces));
    auto ce = cross_entropy(yLogitsL2, idx);
    auto cost = mean(sum(ce, keywords::axis = 2), keywords::axis = 0);

    debug(x, "x");
    debug(gamma, "gamma");
    debug(beta, "beta");

    graph->forward();
    graph->backward();
  }

  /*{
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    auto x = graph->param("x", {batchSize, 3072},
  init=inits::from_vector(temp));
    auto gamma = graph->param("gamma", {1, 3072}, init=inits::from_value(2.0));
    auto beta = graph->param("beta", {1, 3072}, init=inits::zeros);

    auto y = layer_norm(x, gamma, beta);

    auto w = graph->param("w", {3072, 3072}, init=inits::from_vector(temp2));

    auto y2 = tanh(layer_norm(dot(y, w), gamma, beta));

    auto idx = graph->constant({(int)indeces.size(), 1},
                               init=inits::from_vector(indeces));
    auto ce = cross_entropy(y2, idx);
    auto cost = mean(sum(ce, keywords::axis=2), keywords::axis=0);

    debug(x, "x");
    debug(gamma, "gamma");
    debug(beta, "beta");

    graph->forward();
    graph->backward();
  }*/

  return 0;
}
