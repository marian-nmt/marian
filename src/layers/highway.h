#pragma once

#include "layers/generic.h"

namespace marian {

class Motorway : public Layer {
  public:
    Motorway(const std::string name, size_t depth)
      : Layer(name),
        depth_(depth)
    {}

    Expr operator()(Expr x) {
      Expr input = x;
      for (size_t i = 0; i < depth_; ++i) {
        size_t out_dim = x->shape()[1];
        auto g = Dense(name_ + "d1_" + std::to_string(i), out_dim, keywords::activation=act::logit)(x);
        auto dense_2 = Dense(name_ + "d2_" + std::to_string(i), out_dim, keywords::activation=act::linear)(x);
        auto rr = relu(dense_2);
        input = (g * rr) + ((1 - g) * input);
      }
      return input;
    }

  protected:
    size_t depth_;
};

using Highway = Motorway;

}  // namespace marian
