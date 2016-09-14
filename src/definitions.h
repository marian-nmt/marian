#pragma once

#include <vector>
#include <string>
#include <functional>

namespace marian {
  typedef float Float;
  typedef std::vector<int> Shape;
  const int whatevs{-1};
}

#include "keywords.h"
// #include "tensor.h"

namespace marian {
  class Tensor;

  namespace keywords {
    KEY(axis, int)
    KEY(name, std::string)
    KEY(shape, Shape)
    KEY(value, float)
    KEY(lazy_shape, std::function<Shape()>)
    KEY(lazy_value, std::function<float()>)
    KEY(init, std::function<void(Tensor)>)
  }

}
