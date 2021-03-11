// TMP here stands for Template Meta-Programming

#pragma once

#include "functional/array.h"
#include "functional/defs.h"
#include "functional/tensor.h"

namespace marian {
namespace functional {

// This struct and its specializations are never used directly, only through apply and applyWithCast below.
template <size_t K, class Functor, typename AccType> // K-ary application of Functor, elements are cast to AccType before application of Functor
struct FApply {};

template <class Functor, typename AccType>
struct FApply<1, Functor, AccType> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 1>& in,
      const functional::Array<int, 1>& indices) {
    return functor((AccType)in[0].data()[indices[0]]); // indices is an array of offsets into multiple tensors, index[i] corresponds in[i] based on up to arity K
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 1>& in,
      int index) {
    return functor((AccType)in[0].data()[index]);
  }
};

template <class Functor, typename AccType>
struct FApply<2, Functor, AccType> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 2>& in,
      const functional::Array<int, 2>& indices) {
    return functor((AccType)in[0].data()[indices[0]],
                   (AccType)in[1].data()[indices[1]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 2>& in,
      int index) {
    return functor((AccType)in[0].data()[index],
                   (AccType)in[1].data()[index]);
  }
};

template <class Functor, typename AccType>
struct FApply<3, Functor, AccType> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 3>& in,
      const functional::Array<int, 3>& indices) {
    return functor((AccType)in[0].data()[indices[0]],
                   (AccType)in[1].data()[indices[1]],
                   (AccType)in[2].data()[indices[2]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 3>& in,
      int index) {
    return functor((AccType)in[0].data()[index],
                   (AccType)in[1].data()[index],
                   (AccType)in[2].data()[index]);
  }
};

template <class Functor, typename AccType>
struct FApply<4, Functor, AccType> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 4>& in,
      const functional::Array<int, 4>& indices) {
    return functor((AccType)in[0].data()[indices[0]],
                   (AccType)in[1].data()[indices[1]],
                   (AccType)in[2].data()[indices[2]],
                   (AccType)in[3].data()[indices[3]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 4>& in,
      int index) {
    return functor((AccType)in[0].data()[index],
                   (AccType)in[1].data()[index],
                   (AccType)in[2].data()[index],
                   (AccType)in[3].data()[index]);
  }
};

template <class Functor, typename AccType>
struct FApply<5, Functor, AccType> {
  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 5>& in,
      const functional::Array<int, 5>& indices) {
    return functor((AccType)in[0].data()[indices[0]],
                   (AccType)in[1].data()[indices[1]],
                   (AccType)in[2].data()[indices[2]],
                   (AccType)in[3].data()[indices[3]],
                   (AccType)in[4].data()[indices[4]]);
  }

  template <typename ElementType>
  HOST_DEVICE_INLINE static AccType apply(
      Functor functor,
      functional::Array<functional::Tensor<ElementType>, 5>& in,
      int index) {
    return functor((AccType)in[0].data()[index], 
                   (AccType)in[1].data()[index], 
                   (AccType)in[2].data()[index], 
                   (AccType)in[3].data()[index], 
                   (AccType)in[4].data()[index]);
  }
};

/******************************************************************************/
// Applying functor to sets of K tensors
template <typename ElementType, size_t K, class Functor>
HOST_DEVICE_INLINE ElementType apply(Functor functor,
                    functional::Array<functional::Tensor<ElementType>, K>& in,
                    const functional::Array<int, K>& indices) {
  return FApply<K, Functor, ElementType>::apply(functor, in, indices); // functor is applied to same type as input ElementType, no casting required
}

template <typename ElementType, size_t K, class Functor>
HOST_DEVICE_INLINE ElementType apply(Functor functor,
                    functional::Array<functional::Tensor<ElementType>, K>& in,
                    int index) {
  return FApply<K, Functor, ElementType>::apply(functor, in, index); // functor is applied to same type as input ElementType, no casting required
}

template <typename AccType, typename ElementType, size_t K, class Functor>
HOST_DEVICE_INLINE AccType applyWithCast(Functor functor,
                    functional::Array<functional::Tensor<ElementType>, K>& in,
                    const functional::Array<int, K>& indices) {
  return FApply<K, Functor, AccType>::apply(functor, in, indices); // ElementType and AccType are potentially different, cast to AccType before applying functor.
                                                                   // This is useful when accumulating e.g. 16-bit into 32-bit and we want to case to 32-bit before
                                                                   // the functor is applied. L2-Norm is a good use-case since the square can be large. 
}

template <typename AccType, typename ElementType, size_t K, class Functor>
HOST_DEVICE_INLINE AccType applyWithCast(Functor functor,
                    functional::Array<functional::Tensor<ElementType>, K>& in,
                    int index) {
  return FApply<K, Functor, AccType>::apply(functor, in, index); // ElementType and AccType are potentially different, cast to AccType before applying functor
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
      agg = aggFunctor(agg, applyWithCast<AccType>(functor, in, acc));
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
