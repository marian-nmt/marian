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
      //std::cerr << "data_1=" << data_ << std::endl;
    }

    TMatrix(size_t rows, size_t cols, bool zero = false)
    : rows_(rows)
    , cols_(cols)
    , arrSize_(rows * cols)
    {
      std::cerr << "TMatrix2=" << std::endl;
      HANDLE_ERROR( cudaMalloc((void**)&data_, arrSize_ * sizeof(T)) );
      if (zero) {
        std::cerr << "zeros=" << std::endl;
        HANDLE_ERROR( cudaMemset(data_, 0, arrSize_ * sizeof(T)) );
      }
      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "Debug=" << Debug() << std::endl;
    }

    TMatrix(TMatrix&& m)
    : TMatrix()
    {
      std::cerr << "TMatrix1=" << std::endl;
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
      //std::cerr << "data_3=" << data_ << std::endl;
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
          //std::cerr << "newData=" << newData << std::endl;

          HANDLE_ERROR( cudaMemcpyAsync(
              newData,
              data_,
              arrSize_ * sizeof(T),
              cudaMemcpyDeviceToDevice,
              CudaStreamHandler::GetStream()) );

          HANDLE_ERROR(cudaFree(data_));
          //std::cerr << "delete data_1=" << data_ << std::endl;
          data_ = newData;
          arrSize_ = rows * cols;
        }
        else if (rows == 0 || cols == 1) {
          Clear();
        }
      }
      else {
        HANDLE_ERROR( cudaMalloc((void**)&data_, rows * cols * sizeof(T)) );
        //std::cerr << "data_4=" << data_ << " " << (rows * cols) << std::endl;
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
      strm << Rows() << "x" << Cols() << " "
          << data_ << " "
          << arrSize_ << " "
          << std::flush;

      float sum = Sum(data(), size());
      strm << "size=" << size() << " sum=" << sum << std::flush;

      return strm.str();
    }

    void Clear() {
      HANDLE_ERROR(cudaFree(data_));
      //std::cerr << "delete data_2=" << data_ << std::endl;
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


/*
template <typename T>
class TMatrix : public BaseMatrix {
  public:
    typedef DeviceVector<T> VecType;
    typedef typename VecType::value_type value_type;

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
    , arrSize_(size())
    {
      data_ = new VecType(arrSize_);
      if (zero) {
        HANDLE_ERROR( cudaMemset(thrust::raw_pointer_cast(data_->data()), arrSize_ * sizeof(T), 0) );
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
    , arrSize_(size())
    {
      data_ = new VecType(arrSize_);
      HANDLE_ERROR( cudaMemcpyAsync(
          thrust::raw_pointer_cast(data_->data()),
          thrust::raw_pointer_cast(m.data_->data()),
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
        if ((rows * cols) > arrSize_) {
          //HANDLE_ERROR(cudaStreamSynchronize(0));
          VecType *newData = new VecType(rows * cols);
          //HANDLE_ERROR(cudaStreamSynchronize(0));

          HANDLE_ERROR( cudaMemcpyAsync(
              thrust::raw_pointer_cast(newData->data()),
              thrust::raw_pointer_cast(data_->data()),
              size() * sizeof(T),
              cudaMemcpyDeviceToDevice,
              CudaStreamHandler::GetStream()) );

          //HANDLE_ERROR(cudaStreamSynchronize(0));

          delete data_;
          data_ = newData;
          arrSize_ = rows * cols;
        }
        else if (rows == 0 || cols == 1) {
          Clear();
        }
      }
      else {
        data_ = new VecType(rows * cols);
        //HANDLE_ERROR(cudaStreamSynchronize(0));
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
      strm << Rows() << "x" << Cols() << " " << data_ << " ";

      //T sum = 0;
      //for (size_t i = 0; i < size(); ++i) {
      //  sum += (*data_)[i];
      //}
      //strm << "size=" << size() << " sum=" << sum << std::flush;

      if (data_) {
        float sum = Sum(data(), size());
        strm << "size=" << size() << " sum=" << sum << std::flush;
      }
      else {
        strm << "empty array" << std::flush;
      }

      return strm.str();
    }

    void Clear() {
      delete data_;
      data_ = nullptr;
      rows_ = 0;
      cols_ = 0;
      arrSize_ = 0;
    }

    T* data() {
      return thrust::raw_pointer_cast(data_->data());
    }

    const T* data() const {
      return thrust::raw_pointer_cast(data_->data());
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
    VecType *data_;
};
*/
typedef TMatrix<float> Matrix;
typedef TMatrix<int> IMatrix;


}  // namespace mblas
}  // namespace GPU
}
