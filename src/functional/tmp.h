// TMP here stands for Template Meta-Programming

#pragma once

#include "functional/array.h"
#include "functional/defs.h"
#include "functional/tensor.h"

namespace marian {
namespace functional {

template <size_t K, class Functor>
struct FApply {};

template <class Functor>
struct FApply<1, Functor> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 1>& in,
      const functional::Array<int, 1>& indices) {
    return functor(in[0].data()[indices[0]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 1>& in,
      int index) {
    return functor(in[0].data()[index]);
  }
};

template <class Functor>
struct FApply<2, Functor> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 2>& in,
      const functional::Array<int, 2>& indices) {
    return functor(in[0].data()[indices[0]],
                   in[1].data()[indices[1]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 2>& in,
      int index) {
    return functor(in[0].data()[index],
                   in[1].data()[index]);
  }
};

template <class Functor>
struct FApply<3, Functor> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 3>& in,
      const functional::Array<int, 3>& indices) {
    return functor(in[0].data()[indices[0]],
                   in[1].data()[indices[1]],
                   in[2].data()[indices[2]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 3>& in,
      int index) {
    return functor(in[0].data()[index],
                   in[1].data()[index],
                   in[2].data()[index]);
  }
};

template <class Functor>
struct FApply<4, Functor> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 4>& in,
      const functional::Array<int, 4>& indices) {
    return functor(in[0].data()[indices[0]],
                   in[1].data()[indices[1]],
                   in[2].data()[indices[2]],
                   in[3].data()[indices[3]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 4>& in,
      int index) {
    return functor(in[0].data()[index],
                   in[1].data()[index],
                   in[2].data()[index],
                   in[3].data()[index]);
  }
};

template <class Functor>
struct FApply<5, Functor> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 5>& in,
      const functional::Array<int, 5>& indices) {
    return functor(in[0].data()[indices[0]],
                   in[1].data()[indices[1]],
                   in[2].data()[indices[2]],
                   in[3].data()[indices[3]],
                   in[4].data()[indices[4]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static ElementType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 5>& in,
      int index) {
    return functor(in[0].data()[index], 
                   in[1].data()[index], 
                   in[2].data()[index], 
                   in[3].data()[index], 
                   in[4].data()[index]);
  }
};

template <size_t K, class Functor, typename ElementType>
HOST_DEVICE_INLINE ElementType apply(Functor functor,
                    functional::Array<functional::Tensor<ElementType>, K>& in,
                    const functional::Array<int, K>& indices) {
  return FApply<K, Functor>::apply(functor, in, indices);
}

template <size_t K, class Functor, typename ElementType>
HOST_DEVICE_INLINE ElementType apply(Functor functor,
                    functional::Array<functional::Tensor<ElementType>, K>& in,
                    int index) {
  return FApply<K, Functor>::apply(functor, in, index);
}

/******************************************************************************/

// @TODO: Rename this. It is a reduction loop.
template <size_t n, size_t N, size_t K>
struct Loop {
  template <class Functor, class AggFunctor, typename ElementType, typename AccType>
  HOST_DEVICE_INLINE static AccType result(
      Functor functor, AccType aggInit, AggFunctor aggFunctor,
      functional::Array<functional::Tensor<ElementType>, K>& in,
      const functional::Array<int, K>& pAcc,
      const functional::Array<int, N>& length,
      const functional::Array<int, N>& dim) {
    AccType agg = aggInit;
    functional::Array<int, K> acc;
    for(int i = 0; i < length[N - n]; ++i) {
      for(size_t j = 0; j < K; ++j) {
        acc[j] = pAcc[j] + (dim[N - n] + i) * in[j].shape().bstride(N - n);
      }
      agg = aggFunctor(agg, Loop<n - 1, N, K>::result(functor, aggInit, aggFunctor, in, acc, length, dim));
    }
    return agg;
  }
};

template <size_t N, size_t K>
struct Loop<1, N, K> {
  template <class Functor, class AggFunctor, typename ElementType, typename AccType>
  HOST_DEVICE_INLINE static AccType result(
      Functor functor, AccType aggInit, AggFunctor aggFunctor,
      functional::Array<functional::Tensor<ElementType>, K>& in,
      const functional::Array<int, K>& pAcc,
      const functional::Array<int, N>& length,
      const functional::Array<int, N>& dim) {
    AccType agg = aggInit;
    functional::Array<int, K> acc;
    for(int i = 0; i < length[N - 1]; ++i) {
      for(size_t j = 0; j < K; ++j) {
        acc[j] = pAcc[j] + (dim[N - 1] + i) * in[j].shape().bstride(N - 1);
      }
      agg = aggFunctor(agg, (AccType)apply<K>(functor, in, acc));
    }
    return agg;
  }
};


template <size_t N, size_t K, class Functor, class AggFunctor, typename ElementType, typename AccType>
HOST_DEVICE_INLINE AccType loops(Functor functor, AccType aggInit, AggFunctor aggFunctor,
                                     functional::Array<functional::Tensor<ElementType>, K>& in,
                    const functional::Array<int, N>& length,
                    const functional::Array<int, N>& dim) {
  functional::Array<int, K> acc = {0};
  return Loop<N, N, K>::result(functor, aggInit, aggFunctor, in, acc, length, dim);
}
}  // namespace functional
}  // namespace marian
