#include "phoenix_functions.h"

namespace mblas
{
  
  float logit(float x) {
    return 1.0 / (1.0 + expapprox(-x));
  }
  
  float max(float x, float y) {
    return x > y ? x : y;
  }
  
}