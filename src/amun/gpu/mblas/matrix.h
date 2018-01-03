#pragma once

#include <memory>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "common/exception.h"
#include "common/base_matrix.h"
#include "gpu/types-gpu.h"
#include "handles.h"
#include "vector.h"

namespace amunmt {
namespace GPU {
namespace mblas {

using namespace thrust::placeholders;

template <typename T>
class TMatrix : public BaseMatrix {
  public:
    typedef T value_type;

    TMatrix()
    {
      dim_[0] = 0;
      dim_[1] = 0;
      dim_[2] = 0;
      dim_[3] = 0;
    }

    TMatrix(size_t rows, size_t cols, size_t c, size_t d, bool zero = false)
    {
      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = c;
      dim_[3] = d;

      uint newSize = rows * cols * c * d;
      vec_.newSize(newSize);

      if (zero) {
        HANDLE_ERROR( cudaMemsetAsync(vec_.data(), 0, newSize * sizeof(T), CudaStreamHandler::GetStream()) );
      }
    }

    TMatrix(TMatrix&& m)
    : TMatrix()
    {
      swap(m);
    }

    explicit TMatrix(const TMatrix& m)
    : vec_(m.vec_)
    {
      dim_[0] = m.dim_[0];
      dim_[1] = m.dim_[1];
      dim_[2] = m.dim_[2];
      dim_[3] = m.dim_[3];
    }

    ~TMatrix()
    {
    }

    virtual size_t size() const
    {
      return vec_.size();
    }

    virtual size_t dim(size_t i) const
    {
      return dim_[i];
    }

    void Resize(size_t rows, size_t cols, size_t c = 1, size_t d = 1) {
      size_t newSize = cols * rows * c * d;
      vec_.resize(newSize);

      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = c;
      dim_[3] = d;
    }

    void NewSize(size_t rows, size_t cols, size_t c = 1, size_t d = 1) {
      size_t newSize = cols * rows * c * d;
      vec_.newSize(newSize);

      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = c;
      dim_[3] = d;
    }

    virtual std::string Debug(size_t verbosity = 1) const
    {
      std::stringstream strm;
      strm << BaseMatrix::Debug(verbosity) << " ";
      /*
      strm << vec_.data() << " "
          << vec_.maxSize() << " "
          << std::flush;
      */

      if (verbosity) {
        strm << vec_.Debug(verbosity);
      }

      return strm.str();
    }

    value_type* data() {
      return vec_.data();
    }

    const value_type* data() const {
      return vec_.data();
    }

    void swap(TMatrix &other)
    {
      std::swap(dim_, other.dim_);
      vec_.swap(other.vec_);
    }

    TMatrix& operator=(const TMatrix& other)
    {
      if (this != &other) {
        std::copy(other.dim_, other.dim_ + SHAPE_SIZE, dim_);
        vec_ = other.vec_;
      }
      return *this;
    }

  private:
    size_t dim_[SHAPE_SIZE];
    Vector<T> vec_;
};

typedef TMatrix<float> Matrix;


}  // namespace mblas
}  // namespace GPU
}
