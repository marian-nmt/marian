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

__device__
inline half2 htanh(const half2 x)
{
  half2 one = __float2half2_rn(1.0f);
  half2 t1 = h2exp(__hmul2(__float2half2_rn(2.0f), x));
  half2 t2 = __hsub2(one, t1);
  half2 t3 = __hadd2(one, t1);
  t3 = h2rcp(t3);
  half2 ret = __hmul2(t1, t3);

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
      struct half_binary_add : public thrust::binary_function<T,T,T> {
        __host__ __device__
        T operator()(const T &x, const T &y) const { return x + y; }
      };

      template<>
      struct half_binary_add<half2> : public thrust::binary_function<half2,half2,half2> {
        __device__
        half2 operator()(const half2 &x, const half2 &y) const { return __hadd2(x, y); }
      };

      template<typename Eval1, typename Eval2>
      __host__ __device__
      actor<composite<binary_operator<half_binary_add>, actor<Eval1>, actor<Eval2>>>
      HalfAdd(const actor<Eval1> &_1, const actor<Eval2> &_2) {
        return compose(binary_operator<half_binary_add>(), _1, _2);
      }

      //////////////////////////////////////////////////////////////////////////////////////////

    }
  }
}
