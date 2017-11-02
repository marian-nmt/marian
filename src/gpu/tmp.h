#pragma once

#include "gpu/defs.h"
#include "gpu/tensor.h"
#include "gpu/array.h"

namespace marian {
namespace gpu {

template <size_t K, class Functor>
struct FApply {};

template <class Functor>
struct FApply<1, Functor> {
  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 1>& in,
                             const gpu::Array<int, 1>& indices) {
    return functor(in[0][indices[0]]);
  }

  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 1>& in,
                             int index) {
    return functor(in[0][index]);
  }
};

template <class Functor>
struct FApply<2, Functor> {
  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 2>& in,
                             const gpu::Array<int, 2>& indices) {
    return functor(in[0][indices[0]], in[1][indices[1]]);
  }

  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 2>& in,
                             int index) {
    return functor(in[0][index], in[1][index]);
  }
};

template <class Functor>
struct FApply<3, Functor> {
  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 3>& in,
                             const gpu::Array<int, 3>& indices) {
    return functor(in[0][indices[0]], in[1][indices[1]], in[2][indices[2]]);
  }

  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 3>& in,
                             int index) {
    return functor(in[0][index], in[1][index], in[2][index]);
  }
};


template <class Functor>
struct FApply<4, Functor> {
  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 4>& in,
                             const gpu::Array<int, 4>& indices) {
    return functor(in[0][indices[0]],
                   in[1][indices[1]],
                   in[2][indices[2]],
                   in[3][indices[3]]);
  }

  __HDI__ static float apply(Functor functor,
                             gpu::Array<gpu::Tensor<float>, 4>& in,
                             int index) {
    return functor(in[0][index],
                   in[1][index],
                   in[2][index],
                   in[3][index]);
  }
};

template <size_t K, class Functor>
  __HDI__ float apply(Functor functor,
                      gpu::Array<gpu::Tensor<float>, K>& in,
                      const gpu::Array<int, K>& indices) {
  return FApply<K, Functor>::apply(functor, in, indices);
}

template <size_t K, class Functor>
  __HDI__ float apply(Functor functor,
                      gpu::Array<gpu::Tensor<float>, K>& in,
                      int index) {
  return FApply<K, Functor>::apply(functor, in, index);
}

/******************************************************************************/

template <size_t n, size_t N, size_t K>
struct Loop {
  template <class Functor>
  __HDI__ static float result(Functor functor,
                              gpu::Array<gpu::Tensor<float>, K>& in,
                              const gpu::Array<int, K>& pAcc,
                              const gpu::Array<int, N>& length,
                              const gpu::Array<int, N>& dim) {
    float sum = 0;
    gpu::Array<int, K> acc;
    for(int i = 0; i < length[N - n]; ++i) {
      for(int j = 0; j < K; ++j) {
        acc[j] = pAcc[j] + (dim[N - n] + i) * in[j].shape().bstride(N - n);
      }
      sum += Loop<n - 1, N, K>::result(functor, in, acc, length, dim);
    }
    return sum;
  }
};

template <size_t N, size_t K>
struct Loop<1, N, K> {
  template <class Functor>
  __HDI__ static float result(Functor functor,
                              gpu::Array<gpu::Tensor<float>, K>& in,
                              const gpu::Array<int, K>& pAcc,
                              const gpu::Array<int, N>& length,
                              const gpu::Array<int, N>& dim) {
    float sum = 0;
    gpu::Array<int, K> acc;
    for(int i = 0; i < length[N - 1]; ++i) {
      for(int j = 0; j < K; ++j) {
        acc[j] = pAcc[j] + (dim[N - 1] + i) * in[j].shape().bstride(N - 1);
      }
      sum += apply<K>(functor, in, acc);
    }
    return sum;
  }
};

template <size_t N, size_t K, class Functor>
__HDI__ float loops(Functor functor,
                    gpu::Array<gpu::Tensor<float>, K>& in,
                    const gpu::Array<int, N>& length,
                    const gpu::Array<int, N>& dim) {
  gpu::Array<int, K> acc = {0};
  return Loop<N, N, K>::result(functor, in, acc, length, dim);
}

}
}