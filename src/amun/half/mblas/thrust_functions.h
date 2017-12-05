#pragma once

#include <cmath>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

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

namespace thrust
{
  namespace detail
  {
    namespace functional
    {

      //////////////////////////////////////////////////////////////////////////////////////////
      template<typename T>
      struct half_unary_logit : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return (half)1.0 / ((half)1.0 + hexp(-x)); }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<half_unary_logit>, actor<Eval>>>
      HalfLogit(const actor<Eval> &_1) {
        return compose(unary_operator<half_unary_logit>(), _1);
      }

      //////////////////////////////////////////////////////////////////////////////////////////
      template<typename T>
      struct half_unary_tanh : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return htanh(x); }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<half_unary_tanh>, actor<Eval>>>
      HalfTanh(const actor<Eval> &_1) {
        return compose(unary_operator<half_unary_tanh>(), _1);
      }

      //////////////////////////////////////////////////////////////////////////////////////////
      template<typename T>
      struct half_unary_add : public thrust::binary_function<T,T,T> {
        __host__ __device__
        T operator()(const T &x, const T &y) const { return x + y; }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<binary_operator<half_unary_add>, actor<Eval>, actor<Eval>>>
      HalfAdd(const actor<Eval> &_1, const actor<Eval> &_2) {
        return compose(unary_operator<half_unary_add>(), _1, _2);
      }

      //////////////////////////////////////////////////////////////////////////////////////////

    }
  }
}
