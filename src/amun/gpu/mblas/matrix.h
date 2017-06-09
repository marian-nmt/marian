#pragma once

#include <memory>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "common/exception.h"
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
    , beam_(0)
    , batches_(0)
    , arrSize_(0)
    , data_(nullptr)
    {
    }

    TMatrix(size_t rows, size_t cols, size_t beam, size_t batches, bool zero = false)
    : rows_(rows)
    , cols_(cols)
    , beam_(1)
    , batches_(1)
    , arrSize_(size())
    {
      HANDLE_ERROR( cudaMalloc((void**)&data_, arrSize_ * sizeof(T)) );
      //std::cerr << "malloc data1:" << data_ << std::endl;
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
    , beam_(m.beam_)
    , batches_(m.batches_)
    , arrSize_(m.arrSize_)
    {
      HANDLE_ERROR( cudaMalloc((void**)&data_, arrSize_ * sizeof(T)) );
      //std::cerr << "malloc data2:" << data_ << std::endl;
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

    virtual size_t dim(size_t i) const
    {
    	switch (i) {
    	case 0: return rows_;
    	case 1: return cols_;
    	case 2: return beam_;
    	case 3: return batches_;
    	default:
    		abort();
    	}
    }

    void Resize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1) {
      size_t newSize = cols * rows * beam * batches;
      if (data_) {
        if (newSize > arrSize_) {
          T *newData;
          HANDLE_ERROR( cudaMalloc((void**)&newData, newSize * sizeof(T)) );
          //std::cerr << "malloc data3:" << data_ << std::endl;

          //size_t count = std::min(arrSize_, newSize);

          HANDLE_ERROR( cudaMemcpyAsync(
              newData,
              data_,
              size() * sizeof(T),
              cudaMemcpyDeviceToDevice,
              CudaStreamHandler::GetStream()) );

          //std::cerr << "free data1:" << data_ << std::endl;
          HANDLE_ERROR(cudaFree(data_));
          data_ = newData;
          arrSize_ = newSize;
        }
        else if (rows == 0 || cols == 0) {
          Clear();
        }
      }
      else {
        HANDLE_ERROR( cudaMalloc((void**)&data_, newSize * sizeof(T)) );
        //std::cerr << "malloc data4:" << data_ << std::endl;
        arrSize_ = newSize;
      }

      rows_ = rows;
      cols_ = cols;
      beam_ = beam;
      batches_ = batches;
    }

    void Reshape(size_t rows, size_t cols, size_t beam, size_t batches)
    {
      size_t newSize = cols * rows * beam * batches;
      amunmt_UTIL_THROW_IF2(newSize > arrSize_, "Must reshape to same or smaller size");

      rows_ = rows;
      cols_ = cols;
      beam_ = beam;
      batches_ = batches;
    }

    void Reshape2D() {
      rows_ = rows_ * beam_ * batches_;
      beam_ = 1;
      batches_ = 1;
    }

    virtual std::string Debug(size_t verbosity = 1) const
    {
      std::stringstream strm;
      strm << BaseMatrix::Debug(verbosity) << " ";
      strm << data_ << " "
          << arrSize_ << " "
          << std::flush;

      if (verbosity) {
        float sum = Sum(data(), size());
        strm << "sum=" << sum << std::flush;

        if (verbosity == 2) {
          cudaStream_t& stream = CudaStreamHandler::GetStream();
          T h_data[size()];

          HANDLE_ERROR( cudaMemcpyAsync(
              &h_data,
              data_,
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

    void Clear() {
      //std::cerr << "free data2:" << data_ << std::endl;
      HANDLE_ERROR(cudaFree(data_));
      data_ = nullptr;
      rows_ = 0;
      cols_ = 0;
      beam_ = 0;
      batches_ = 0;
      arrSize_ = 0;
    }

    value_type* data() {
      return data_;
    }

    const value_type* data() const {
      return data_;
    }

    void swap(TMatrix &other)
    {
      std::swap(rows_, other.rows_);
      std::swap(cols_, other.cols_);
      std::swap(beam_, other.beam_);
      std::swap(batches_, other.batches_);
      std::swap(arrSize_, other.arrSize_);
      std::swap(data_, other.data_);
    }

  private:
    size_t rows_;
    size_t cols_;
    size_t beam_;
    size_t batches_;
    size_t arrSize_;
    T *data_;
};

typedef TMatrix<float> Matrix;
typedef TMatrix<int> IMatrix;


}  // namespace mblas
}  // namespace GPU
}
