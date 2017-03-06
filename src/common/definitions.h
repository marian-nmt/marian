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

#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "shape.h"
#include "common/logging.h"

namespace marian {

  template<class T>
  using Ptr = std::shared_ptr<T>;

  template<class T>
  using UPtr = std::unique_ptr<T>;

  /** @brief Creates shared_ptr of any type, passes all arguments to any available constructor */
  template <class T, typename ...Args>
  Ptr<T> New(Args&& ... args) {
    return Ptr<T>(new T(std::forward<Args>(args)...));
  }

  template <class T>
  Ptr<T> New(Ptr<T> p) {
    return Ptr<T>(p);
  }

  typedef float Float;

  template<class T>
  using DeviceVector = thrust::device_vector<T>;

  template<class T>
  using HostVector = thrust::host_vector<T>;

  /** @brief A placeholder that represents the size of a dimension, the actual value of which is to be specified at some later point.
   *
   * For example, in certain cases the value of one dimension in a Shape object may be used to represent batch size.
   * In such a case, the value of batch size may not be known when the Shape object is constructed.
   * In that case, this placeholder would be used to specify that the batch size value will be defined at some later point.
   */
  const int whatevs{-1};
}


#include "keywords.h"

namespace marian {

  class TensorBase;
  typedef Ptr<TensorBase> Tensor;

  template <class DataType> class Chainable;
  typedef Ptr<Chainable<Tensor>> Expr;

  class OptimizerBase;
  typedef Ptr<OptimizerBase> OptimizerBasePtr;

  class ClipperBase;
  typedef Ptr<ClipperBase> ClipperBasePtr;

  class RunBase;
  typedef Ptr<RunBase> RunBasePtr;

  // An enumeration of activations
  enum struct act { linear, tanh, logit, ReLU };

  // An enumeration of directions
  enum struct dir { forward, backward, bidirect };

  /**
   * @brief Defines a set of keywords.
   *
   * Each invocation of the KEY(name, value_type) macro
   *    will result in the creation of an instance of the Keyword class.
   */
  namespace keywords {
    KEY(axis, int);
    KEY(shape, Shape);
    KEY(value, float);
    KEY(prefix, std::string);
    KEY(final, bool);
    KEY(output_last, bool);
    KEY(activation, act);
    KEY(direction, dir);
    KEY(mask, Expr);
    KEY(dropout_prob, float);
    KEY(init, std::function<void(Tensor)>);


    KEY(eta, float);
    KEY(beta1, float);
    KEY(beta2, float);
    KEY(eps, float);
    KEY(optimizer, Ptr<OptimizerBase>);
    KEY(clip, Ptr<ClipperBase>);
    KEY(batch_size, int);
    KEY(normalize, bool);
    KEY(skip, bool);
    KEY(skip_first, bool);
    KEY(coverage, Expr);
    KEY(max_epochs, int);
    KEY(valid, Ptr<RunBase>);
  }

}
