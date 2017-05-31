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

          //size_t count = std::min(arrSize_, newSize);

          HANDLE_ERROR( cudaMemcpyAsync(
              newData,
              data_,
              size() * sizeof(T),
              cudaMemcpyDeviceToDevice,
              CudaStreamHandler::GetStream()) );

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


////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class TMatrixWrapper
{
public:
  TMatrixWrapper(const TMatrix<T> &matrix)
  {
    dim_[0] = matrix.dim(0);
    dim_[1] = matrix.dim(1);
    dim_[2] = matrix.dim(2);
    dim_[3] = matrix.dim(3);

    data_ = nullptr;
    dataConst_ = matrix.data();
  }

  TMatrixWrapper(TMatrix<T> &matrix)
  {
    dim_[0] = matrix.dim(0);
    dim_[1] = matrix.dim(1);
    dim_[2] = matrix.dim(2);
    dim_[3] = matrix.dim(3);

    data_ = matrix.data();
    dataConst_ = data_;
  }

  TMatrixWrapper(const size_t *dim)
  { // test constructor
    dim_[0] = dim[0];
    dim_[1] = dim[1];
    dim_[2] = dim[2];
    dim_[3] = dim[3];
  }

  __device__ __host__
  size_t dim(size_t i) const
  {  return dim_[i]; }

  __device__ __host__
  size_t size(size_t maxDim = SHAPE_SIZE) const
  {
    //assert(maxDim >= 1);

    size_t ret = 1;
    for (size_t i = 0; i < maxDim; ++i) {
      ret *= dim_[i];
    }
    return ret;
  }

  __device__
  T* data()
  {
    assert(data_);
    return data_;
  }

  __device__
  const T* data() const
  {
    assert(dataConst_);
    return dataConst_;
  }

  __device__
  const T &operator[](size_t i) const
  {
    assert(i < size());
    return data()[i];
  }

  __device__
  T &operator[](size_t i)
  {
  	assert(i < size());
    return data()[i];
  }

  __device__
  const T &get(size_t indices[SHAPE_SIZE]) const
  {
    size_t id = indices2Id(indices);
    assert(id < size());
    return data()[id];
  }

  /*
  __device__
  T &operator[](size_t indices[SHAPE_SIZE])
  {
    size_t id = indices2Id(indices);
    return data()[id];
  }
  */
  __device__ __host__
  void id2MatrixInd(size_t id, size_t out[SHAPE_SIZE]) const
  {
    assert(id < size());

    for (size_t i = 0; i < SHAPE_SIZE; ++i) {
      size_t thisSize = size(i);
      size_t ind = (id / thisSize) % dim_[i];
      assert(ind < dim_[i]);

      out[i] = ind;
    }
  }

  __device__ __host__
  size_t indices2Id(size_t indices[SHAPE_SIZE]) const
  {
    size_t ind = 0;
    for (size_t i = 0; i < SHAPE_SIZE; ++i) {
      ind += indices[i] * size(i);
    }
    assert(ind < size());
    return ind;
  }

protected:
  size_t dim_[SHAPE_SIZE];

  T *data_;
  const T *dataConst_;

};

///////////////////////////////////////////////////////////////////////////////

inline void testidToMatrixInd()
{
  size_t dim[4] = {2, 4, 3, 5};
  TMatrixWrapper<float> matrix(dim);

  std::cerr << "size=" << matrix.size() << std::endl;
  std::cerr << "size0=" << matrix.size(0) << std::endl;
  std::cerr << "size1=" << matrix.size(1) << std::endl;
  std::cerr << "size2=" << matrix.size(2) << std::endl;
  std::cerr << "size3=" << matrix.size(3) << std::endl;
  std::cerr << "size4=" << matrix.size(4) << std::endl;

  for (size_t i = 0; i < matrix.size(); ++i) {
    matrix.id2MatrixInd(i, dim);

    std::cerr << i << "=";
    for (size_t j = 0; j < SHAPE_SIZE; ++j) {
      std::cerr << " " << dim[j];
    }

    std::cerr << " = " << matrix.indices2Id(dim);
    std::cerr << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////


}  // namespace mblas
}  // namespace GPU
}
