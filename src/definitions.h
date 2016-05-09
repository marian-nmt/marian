#pragma once

#include <vector>
#include <string>
#include <functional>

namespace marian {
typedef float Float;
}

#include "keywords.h"
#include "tensor.h"

namespace marian {
  
typedef std::vector<int> Shape;
const int whatevs{-1};

namespace keywords {
  KEY(init, std::function<void(Tensor)>)
  KEY(axis, int)
  KEY(name, std::string)
  KEY(shape, Shape)
  KEY(value, float)
  KEY(lazy_shape, std::function<Shape()>)
  KEY(lazy_value, std::function<float()>)
}

}
