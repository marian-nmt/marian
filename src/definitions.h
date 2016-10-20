#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
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

#include "shape.h"

namespace marian {
  /** @brief Creates shared_ptr of any type, passes all arguments to any available constructor */
  template <class T, typename ...Args>
  std::shared_ptr<T> New(Args&& ... args) {
    return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
  }


  typedef float Float;

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
  typedef std::unique_ptr<TensorBase> Tensor;

  class OptimizerBase;
  typedef std::shared_ptr<OptimizerBase> OptimizerBasePtr;

  class RunBase;
  typedef std::shared_ptr<RunBase> RunBasePtr;

  /**
   * @brief Defines a set of keywords.
   *
   * Each invocation of the KEY(name, value_type) macro
   *    will result in the creation of an instance of the Keyword class.
   */
  namespace keywords {
    KEY(axis, int)
    //KEY(name, std::string)
    KEY(shape, Shape)
    KEY(no_inference, bool)
    KEY(no_training, bool)
    KEY(value, float)
    KEY(lazy_shape, std::function<Shape()>)
    KEY(lazy_value, std::function<float()>)
    KEY(init, std::function<void(Tensor&)>)

    KEY(optimizer, OptimizerBasePtr)
    KEY(batch_size, int)
    KEY(max_epochs, int)
    KEY(valid, RunBasePtr)
  }

}
