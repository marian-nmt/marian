#pragma once

#include "functional/defs.h"

namespace marian {
namespace functional {

// approximate any unary float function within range with
// piecewise linear functions in equal steps. 
//
// Example:
// static Approx<10, 0, 100> approxSigmoid(stableSigmoid);
// float y = approxSigmoid(x);
//
// Creates a functor for range [-10,10] with piecewise linear
// approximations of a sigmoid, 100 pieces, step 0.2.
// This is quite fast on the CPU. 
//
// approxSigmoid.grad(x) computes the corresponding gradient. 
// 
// When used as a local variable, use static keyword to create 
// only once. 

template <int radius = 5, int offset = 0, int pieces = 10>
struct Approx {
  float a[pieces + 2];
  float b[pieces + 2];


  template <typename Function>
  Approx(const Function& function) {
    for(int i = 1; i <= pieces; ++i) {
      float x0 = domain(i - 1);
      float x1 = domain(i);
      float y0 = function(x0);
      float y1 = function(x1);

      a[i] = (y1 - y0) / (x1 - x0);
      b[i] = y0 - a[i] * x0;

      // std::cerr << x0 << " " << x1 << " : " << a[i] << " " << b[i] << std::endl;
      // std::cerr << y0 << " " << y1 << " - " << a[i] * x0 + b[i] << " " << a[i] * x1 + b[i] << std::endl;
    }
    a[0] = 0;
    b[0] = function(domain(0));

    a[pieces + 1] = 0;
    b[pieces + 1] = function(domain(pieces));

    // std::cerr << std::endl << radius << " " << pieces << std::endl;
    // for(int i = 0; i < 100; i++) {
    //   float x = -6.f + i * (12.f/100);
    //   std::cerr << x << " -> " << index(x) << " " << operator()(x) << " " << function(x) << std::endl;
    // }

  }

  __HDI__ int index(float x) const {
    if(x <= -radius)
      return 0;
    if(x < radius)  // +1 because 0 holds value for x < -radius
      return int((x + radius - offset) / ((2.f * radius) / pieces) + 1);
    return pieces + 1;
  }

  __HDI__ float domain(int i) const {
    return i * ((2.f * radius) / pieces) + offset - radius;
  }

  __HDI__ float operator()(float x) const {
    int i = index(x);
    return a[i] * x + b[i];
  }

  __HDI__ float grad(float x) const {
    int i = index(x);
    return a[i];
  }
};

}  // namespace functional
}  // namespace marian
