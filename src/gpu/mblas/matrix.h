#pragma once

#include <memory>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "common/base_matrix.h"
#include "gpu/types-gpu.h"
#include "handles.h"

namespace amunmt {
namespace GPU {
namespace mblas {

using namespace thrust::placeholders;

float Sum(const float *data, size_t count);


template <typename T>
class TMatrix : public BaseMatrix {
  public:
    typedef T value_type;

    TMatrix()
    : rows_(0)
    , cols_(0)
    , arrSize_(0)
    , data_(nullptr)
    {
    }

    TMatrix(size_t rows, size_t cols, bool zero = false)
    : rows_(rows)
    , cols_(cols)
    , arrSize_(rows * cols)
    {
      HANDLE_ERROR( cudaMalloc((void**)&data_, arrSize_ * sizeof(T)) );
      if (zero) {
        HANDLE_ERROR( cudaMemset(data_, 0, arrSize_ * sizeof(T)) );
      }
    }

    TMatrix(TMatrix&& m)
    : TMatrix()
    {
      swap(m);
    }

    TMatrix(const TMatrix& m)
    : rows_(m.rows_)
    , cols_(m.cols_)
    , arrSize_(m.arrSize_)
    {
      HANDLE_ERROR( cudaMalloc((void**)&data_, arrSize_ * sizeof(T)) );
      HANDLE_ERROR( cudaMemcpyAsync(
          data_,
          m.data_,
          arrSize_ * sizeof(T),
          cudaMemcpyDeviceToDevice,
          CudaStreamHandler::GetStream()) );
    }

    ~TMatrix()
    {
      Clear();
    }

    size_t Rows() const {
      return rows_;
    }

    size_t Cols() const {
      return cols_;
    }

    void Resize(size_t rows, size_t cols) {
      if (data_) {
        if ((cols*rows) > arrSize_) {
          T *newData;
          HANDLE_ERROR( cudaMalloc((void**)&newData, rows * cols * sizeof(T)) );

          HANDLE_ERROR( cudaMemcpyAsync(
              newData,
              data_,
              arrSize_ * sizeof(T),
              cudaMemcpyDeviceToDevice,
              CudaStreamHandler::GetStream()) );

          HANDLE_ERROR(cudaFree(data_));
          data_ = newData;
          arrSize_ = rows * cols;
        }
        else if (rows == 0 || cols == 1) {
          Clear();
        }
      }
      else {
        HANDLE_ERROR( cudaMalloc((void**)&data_, rows * cols * sizeof(T)) );
        arrSize_ = rows * cols;
      }
      rows_ = rows;
      cols_ = cols;
    }

    void Reshape(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
    }

    virtual std::string Debug() const
    {
      std::stringstream strm;
      strm << BaseMatrix::Debug() << " ";
      strm << data_ << " "
          << arrSize_ << " "
          << std::flush;

      float sum = Sum(data(), size());
      strm << "size=" << size() << " sum=" << sum << std::flush;

      return strm.str();
    }

    void Clear() {
      HANDLE_ERROR(cudaFree(data_));
      data_ = nullptr;
      rows_ = 0;
      cols_ = 0;
      arrSize_ = 0;
    }

    value_type* data() {
      return data_;
    }

    const value_type* data() const {
      return data_;
    }

    size_t size() const {
      // return data_.size();
      return cols_ * rows_;
    }

    void swap(TMatrix &other)
    {
      std::swap(rows_, other.rows_);
      std::swap(cols_, other.cols_);
      std::swap(arrSize_, other.arrSize_);
      std::swap(data_, other.data_);
    }

  private:
    size_t rows_;
    size_t cols_;
    size_t arrSize_;
    T *data_;
};

typedef TMatrix<float> Matrix;
typedef TMatrix<int> IMatrix;


}  // namespace mblas
}  // namespace GPU
}
