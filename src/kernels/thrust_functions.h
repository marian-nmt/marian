#pragma once

// This file is part of the Marian toolkit.

//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cmath>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

namespace thrust
{
  namespace detail
  {
    namespace functional
    {

      template<typename T>
      struct unary_exp : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const {
            return expf(x);
          }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_exp>, actor<Eval>>>
      Exp(const actor<Eval> &_1) {
        return compose(unary_operator<unary_exp>(), _1);
      }


      template<typename T>
      struct unary_log : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const {
          return logf(x);
        }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_log>, actor<Eval>>>
      Log(const actor<Eval> &_1) {
        return compose(unary_operator<unary_log>(), _1);
      }

      template<typename T>
      struct unary_sigma : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const {
          return 1.0 / (1.0 + expf(-x));
        }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_sigma>, actor<Eval>>>
      Sigma(const actor<Eval> &_1) {
        return compose(unary_operator<unary_sigma>(), _1);
      }

      template<typename T>
      struct unary_tanh : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return tanhf(x); }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_tanh>, actor<Eval>>>
      Tanh(const actor<Eval> &_1) {
        return compose(unary_operator<unary_tanh>(), _1);
      }

      template<typename T>
      struct unary_sqrt : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return sqrtf(x); }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_sqrt>, actor<Eval>>>
      Sqrt(const actor<Eval> &_1) {
        return compose(unary_operator<unary_sqrt>(), _1);
      }

      template<typename T1, typename T2>
      __host__ __device__
      actor<composite<binary_operator<thrust::maximum>, actor<T1>, actor<T2>>>
      Max(const actor<T1> &_1, const actor<T2> &_2) {
        return compose(binary_operator<thrust::maximum>(),
                       make_actor(_1),
                       make_actor(_2));
      }

      template<typename T>
      struct unary_relu : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return x > 0.0f ? x : 0.0f; }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_relu>, actor<Eval>>>
      ReLU(const actor<Eval> &_1) {
        return compose(unary_operator<unary_relu>(), _1);
      }

      template<typename T>
      struct unary_reluback : public thrust::unary_function<T,T> {
        __host__ __device__
        T operator()(const T &x) const { return x > 0.0f ? 1.0f : 0.0f; }
      };

      template<typename Eval>
      __host__ __device__
      actor<composite<unary_operator<unary_reluback>, actor<Eval>>>
      ReLUback(const actor<Eval> &_1) {
        return compose(unary_operator<unary_reluback>(), _1);
      }

      template <typename T>
      __host__ __device__
      int sgn(T val) {
        return (float(0) < val) - (val < float(0));
      }

      template<typename T>
      struct binary_clip : public thrust::binary_function<T, T, T> {
        __host__ __device__
        T operator()(const T &x, const T &y) const { return abs(x) >= y ? sgn(x) * y : x; }
      };

      template<typename T1, typename T2>
      __host__ __device__
      actor<
        composite<
          binary_operator<binary_clip>,
          actor<T1>,
          typename as_actor<T2>::type
        >
      >
      Clip(const actor<T1> &_1, const T2 &_2)
      {
        return compose(binary_operator<binary_clip>(),
                       make_actor(_1),
                       make_actor(_2));
      }

      template<typename T>
      struct binary_pow : public thrust::binary_function<T, T, T> {
        __host__ __device__
        T operator()(const T &x, const T &y) const {
          float tx = x;
          if(y == (int)y && (int)y % 2 == 0)
            tx = abs(x);
          return powf(tx, y);
        }
      };

      template<typename T1, typename T2>
      __host__ __device__
      actor<
        composite<
          binary_operator<binary_pow>,
          actor<T1>,
          typename as_actor<T2>::type
        >
      >
      Pow(const actor<T1> &_1, const T2 &_2)
      {
        return compose(binary_operator<binary_pow>(),
                       make_actor(_1),
                       make_actor(_2));
      }
    }
  }
}
