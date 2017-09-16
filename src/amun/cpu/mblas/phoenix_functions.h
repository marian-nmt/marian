#pragma once

#include "simd_math_prims.h"
#include <math.h>

namespace amunmt {
namespace CPU {
namespace mblas {

inline float exp(float x) {
#ifdef EXACT_MODE
  return std::exp(x);
#else
  return expapprox(x);
#endif
}

inline float log(float x) {
#ifdef EXACT_MODE
  return std::log(x);
#else
  return logapprox(x);
#endif
}

inline float logit(float x) {
#ifdef EXACT_MODE
  return 1.0f / (1.0f + exp(-x));
#else
  return logitapprox(x);
#endif
}

inline float tanh(float x) {
#ifdef EXACT_MODE
  return std::tanh(x);
#else
  return tanhapprox(x);
#endif
}


struct Exp {
  template <typename T>
  inline T operator()(T val) const {
    return exp(val);
  }
};

struct Log {
  template <typename T>
  inline T operator()(T val) const {
    return log(val);
  }
};

struct Logit {
  template <typename T>
  inline T operator()(T val) const {
    return logit(val);
  }
};

struct Tanh {
  template <typename T>
  inline T operator()(T val) const {
    return tanh(val);
  }
};

}
}
}
