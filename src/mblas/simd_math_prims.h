/*

The MIT License (MIT)

Copyright (c) 2015 Jacques-Henri Jourdan <jourgun@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef SIMD_MATH_PRIMS_H
#define SIMD_MATH_PRIMS_H

#include<math.h>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

/* Workaround a lack of optimization in gcc */
const float exp_cst1 = 2139095040.f;
const float exp_cst2 = 0.f;

/* Relative error bounded by 1e-5 for normalized outputs
   Returns invalid outputs for nan inputs
   Continuous error */
inline float expapprox(float val) {
  union { int i; float f; } xu, xu2;
  float val2, val3, val4, b;
  int val4i;
  val2 = 12102203.1615614f*val+1065353216.f;
  val3 = val2 < exp_cst1 ? val2 : exp_cst1;
  val4 = val3 > exp_cst2 ? val3 : exp_cst2;
  val4i = (int) val4;
  xu.i = val4i & 0x7F800000;
  xu2.i = (val4i & 0x7FFFFF) | 0x3F800000;
  b = xu2.f;

  /* Generated in Sollya with:
     > f=remez(1-x*exp(-(x-1)*log(2)),
               [|1,(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x|],
               [1,2], exp(-(x-1)*log(2)));
     > plot(exp((x-1)*log(2))/(f+x)-1, [1,2]);
     > f+x;
  */
  return
    xu.f * (0.510397365625862338668154f + b *
            (0.310670891004095530771135f + b *
             (0.168143436463395944830000f + b *
              (-2.88093587581985443087955e-3f + b *
               1.3671023382430374383648148e-2f))));
}

/* Absolute error bounded by 1e-6 for normalized inputs
   Returns a finite number for +inf input
   Returns -inf for nan and <= 0 inputs.
   Continuous error. */
inline float logapprox(float val) {
  union { float f; int i; } valu;
  float exp, addcst, x;
  valu.f = val;
  exp = valu.i >> 23;
  /* 89.970756366f = 127 * log(2) - constant term of polynomial */
  addcst = val > 0 ? -89.970756366f : -(float)INFINITY;
  valu.i = (valu.i & 0x7FFFFF) | 0x3F800000;
  x = valu.f;


  /* Generated in Sollya using :
    > f = remez(log(x)-(x-1)*log(2),
            [|1,(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x,
              (x-1)*(x-2)*x*x*x|], [1,2], 1, 1e-8);
    > plot(f+(x-1)*log(2)-log(x), [1,2]);
    > f+(x-1)*log(2)
 */
  return
    x * (3.529304993f + x * (-2.461222105f +
      x * (1.130626167f + x * (-0.288739945f +
        x * 3.110401639e-2f))))
    + (addcst + 0.69314718055995f*exp);
}

inline float logitapprox(float x) {
  return 1.0f / (1.0f + expapprox(-x));
}

inline float tanhapprox(float x) {
  x = std::max(std::min(x, 4.97f), -4.97f);
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return a / b;
}


#ifdef __cplusplus
}
#endif

#endif
