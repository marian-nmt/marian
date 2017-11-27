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

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void gSum(const T *data, size_t count, T &ret)
{
  ret = 0;
  for (size_t i = 0; i < count; ++i) {
    ret += data[i];
  }
}

template<typename T>
T Sum(const T *data, size_t count)
{
  T ret;
  T *d_ret;
  HANDLE_ERROR( cudaMalloc(&d_ret, sizeof(T)) );

  const cudaStream_t stream = CudaStreamHandler::GetStream();

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  gSum<<<1, 1, 0, stream>>>(data, count, *d_ret);
  HANDLE_ERROR( cudaMemcpyAsync(&ret, d_ret, sizeof(T), cudaMemcpyDeviceToHost, stream) );

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaFree(d_ret));

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

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

    TMatrix(const TMatrix& m)
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
      strm << vec_.data() << " "
          << vec_.size() << " "
          << vec_.maxSize() << " "
          << std::flush;

      if (verbosity) {
        T sum = Sum(data(), size());
        strm << "sum=" << sum << std::flush;

        if (verbosity == 2) {
          const cudaStream_t& stream = CudaStreamHandler::GetStream();
          T h_data[size()];

          HANDLE_ERROR( cudaMemcpyAsync(
              &h_data,
              vec_.data(),
              size() * sizeof(T),
              cudaMemcpyDeviceToHost,
              stream) );
          HANDLE_ERROR( cudaStreamSynchronize(stream) );

          for (size_t i = 0; i < size(); ++i) {
            strm << " " << h_data[i];
          }
        }
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

  private:
    size_t dim_[SHAPE_SIZE];
    Vector<T> vec_;
};

typedef TMatrix<float> Matrix;
typedef TMatrix<uint> IMatrix;


}  // namespace mblas
}  // namespace GPU
}
