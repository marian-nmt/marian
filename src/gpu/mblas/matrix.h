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
      std::cerr << "data_1=" << data_ << std::endl;
    }

    TMatrix(size_t rows, size_t cols, bool zero = false)
    : rows_(rows)
    , cols_(cols)
    , arrSize_(rows * cols)
    {
      HANDLE_ERROR( cudaMalloc((void**)&data_, arrSize_ * sizeof(T)) );
      if (zero) {
        HANDLE_ERROR( cudaMemset(data_, arrSize_ * sizeof(T), 0) );
      }
      //HANDLE_ERROR(cudaStreamSynchronize(0));
      std::cerr << "data_2=" << data_ << std::endl;
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
      std::cerr << "data_3=" << data_ << std::endl;
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
          std::cerr << "newData=" << newData << std::endl;

          HANDLE_ERROR( cudaMemcpyAsync(
              newData,
              data_,
              arrSize_ * sizeof(T),
              cudaMemcpyDeviceToDevice,
              CudaStreamHandler::GetStream()) );

          HANDLE_ERROR(cudaFree(data_));
          std::cerr << "delete data_1=" << data_ << std::endl;
          data_ = newData;
          arrSize_ = rows * cols;
        }
      }
      else {
        HANDLE_ERROR( cudaMalloc((void**)&data_, rows * cols * sizeof(T)) );
        std::cerr << "data_4=" << data_ << " " << (rows * cols) << std::endl;
        arrSize_ = rows * cols;
        //HANDLE_ERROR(cudaStreamSynchronize(0));
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
      HANDLE_ERROR(cudaStreamSynchronize(0));
      std::stringstream strm;
      std::cerr << Rows() << "x" << Cols() << " "
          << data_ << " "
          << arrSize_ << " "
          << std::flush;

      float sum = Sum(data_, size());
      std::cerr << "sum=" << sum << std::flush;

      return strm.str();
    }

    void Clear() {
      HANDLE_ERROR(cudaFree(data_));
      std::cerr << "delete data_2=" << data_ << std::endl;
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
