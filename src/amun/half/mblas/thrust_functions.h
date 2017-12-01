#pragma once

#include "gpu/mblas/thrust_functions.h"

#define __fp16 half

__fp16 float2half_rn (float a);

__device__
inline half htanh(const half x)
{
  //half ret = ((half)1.0f - hexp((half)-2.0f * x)) / ((half)1.0f + hexp((half)-2.0f * x));
  //half ret = (hexp((half)2.0f * x) - (half)1.0f) / (hexp((half)2.0f * x) + (half)1.0f);
  //half ret = (hexp(x) - hexp(-x)) / (hexp(x) + hexp(-x));
  half ret = tanhf(x);

  return ret;
}
