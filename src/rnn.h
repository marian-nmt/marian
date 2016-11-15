#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"

namespace marian {

struct ParametersTanh {
  Expr U, W, b;
  float dropout = 0;
};

class Tanh {
  public:
    Tanh(const ParametersTanh& params)
    : params_(params) {}

    Expr apply(Expr input, Expr state) {
      Expr output = dot(input, params_.W) + dot(state, params_.U);
      if(params_.b)
        output += params_.b;
      output = tanh(output);

      if(params_.dropout > 0)
        output = dropout(output, value=params_.dropout);

      return output;
    }

  private:
    const ParametersTanh& params_;
};

struct ParametersGRU {
  Expr Uz, Wz, bz;
  Expr Ur, Wr, br;
  Expr Uh, Wh, bh;
  float dropout = 0;
};

class GRU {
  public:
    GRU(const ParametersGRU& params)
    : params_(params) {}

    Expr apply(Expr input, Expr state) {
      using namespace keywords;

      Expr z = dot(input, params_.Wz) + dot(state, params_.Uz);
      if(params_.bz)
        z += params_.bz;
      z = logit(z);

      Expr r = dot(input, params_.Wr) + dot(state, params_.Ur);
      if(params_.br)
        r += params_.br;
      r = logit(r);

      Expr h = dot(input, params_.Wh) + dot(state, params_.Uh) * r;
      if(params_.bh)
        h += params_.bh;
      h = tanh(h);

      // constant 1 in (1-z)*h+z*s
      auto one = state->graph()->ones(shape=state->shape());

      auto output = (one - z) * h + z * state;

      if(params_.dropout > 0)
        output = dropout(output, value=params_.dropout);

      return output;
    }

  private:
    const ParametersGRU& params_;
};

template <class Cell = Tanh>
class RNN {
  public:

    template <class Parameters>
    RNN(const Parameters& params)
    : cell_(params) {}

    std::vector<Expr> apply(const std::vector<Expr>& inputs,
                            const Expr initialState) {
      std::vector<Expr> outputs;
      auto state = initialState;
      for(auto input : inputs) {
        state = cell_.apply(input, state);
        outputs.push_back(state);
      }
      return outputs;
    }

  private:
    Cell cell_;
};

}
