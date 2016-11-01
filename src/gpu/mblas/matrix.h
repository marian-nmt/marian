#pragma once

#include <cmath>
#include <memory>
#include <sstream>

#include "common/base_matrix.h"

#define MAX_THREADS 512
#define MAX_BLOCKS 65535

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#ifdef __APPLE__
#include <boost/thread/tss.hpp>
#include <boost/pool/object_pool.hpp>
#endif

#include "thrust_functions.h"
#include "common/god.h"
#include "common/exception.h"
#include "common/hypothesis.h"
#include "common/soft_alignment.h"

#include "gpu/decoder/encoder_decoder.h"
#include "gpu/types-gpu.h"
#include "gpu/nth_element.h"


namespace GPU {
namespace mblas {

using namespace thrust::placeholders;

struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  __host__ __device__
  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};


template <class VecType>
class TMatrix : public BaseMatrix {
  public:
    typedef typename VecType::value_type value_type;
    typedef typename VecType::iterator iterator;
    typedef typename VecType::const_iterator const_iterator;

    TMatrix()
    : rows_(0), cols_(0)
    {}

    TMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows_ * cols_)
    {}

    TMatrix(size_t rows, size_t cols, value_type val)
    : rows_(rows), cols_(cols), data_(rows_ * cols_, val)
    {}

    TMatrix(TMatrix&& m)
    : rows_(m.rows_), cols_(m.cols_), data_(std::move(m.data_)) {}

    TMatrix(const TMatrix& m) = delete;

    value_type operator()(size_t i, size_t j) const {
      return data_[i * cols_ + j];
    }

    void Set(size_t i, size_t j, float value)  {
      data_[i * cols_ + j] = value;
    }

    size_t Rows() const {
      return rows_;
    }

    size_t Cols() const {
      return cols_;
    }

    void Resize(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
      data_.resize(rows_ * cols_);
    }

    void Resize(size_t rows, size_t cols, value_type val) {
      rows_ = rows;
      cols_ = cols;
      data_.resize(rows_ * cols_, val);
    }

    void Reserve(size_t rows, size_t cols) {
      data_.reserve(rows * cols);
    }

    void Reshape(size_t rows, size_t cols) {
      rows_ = rows;
      cols_ = cols;
    }

    virtual std::string Debug() const
    {
      std::stringstream strm;
      strm << Rows() << "x" << Cols() << ":";
      for (size_t row = 0; row < Rows(); ++row) {
        float rowSum = 0;
        for (size_t col = 0; col < Cols(); ++col) {
          rowSum += (*this)(row, col);
        }
        strm << rowSum << " ";
      }
      return strm.str();
    }

    void Purge() {
      Clear();
      VecType temp;
      data_.swap(temp);
    }

    void Clear() {
      data_.clear();
      rows_ = 0;
      cols_ = 0;
    }

    VecType& GetVec() {
      return data_;
    }


    const VecType& GetVec() const {
      return data_;
    }

    value_type* data() {
      return thrust::raw_pointer_cast(data_.data());
    }

    const value_type* data() const {
      return thrust::raw_pointer_cast(data_.data());
    }

    iterator begin() {
      return data_.begin();
    }

    iterator end() {
      return data_.end();
    }

    const_iterator begin() const{
      return data_.begin();
    }

    const_iterator end() const {
      return data_.end();
    }

    size_t size() const {
      return data_.size();
    }

  static cudaStream_t& GetStream() {
    if (stream_ == nullptr) {
      assert(stream_ == nullptr);
      stream_ = new cudaStream_t;
      // cudaStreamCreateWithFlags(stream_, cudaStreamNonBlocking);
      cudaStreamCreate(stream_);
    }
    return *stream_;
  }

  private:
    size_t rows_;
    size_t cols_;
    VecType data_;
    static thread_local cudaStream_t* stream_;
};

typedef thrust::device_vector<float> FVec;
typedef thrust::device_vector<unsigned int> IVec;

class CublasHandler {
  public:
    static cublasHandle_t GetHandle() {
#ifdef __APPLE__
      cublasHandle_t *handle = handle_.get();
      if (handle == nullptr) {
        handle = new cublasHandle_t;
        handle_.reset(handle);
      }
      return *handle;
#else
      if(handle_ == nullptr) {
      assert(handle_ == nullptr);
      handle_ = new cublasHandle_t;
      cublasCreate(handle_);
      cublasSetStream(*handle_, Matrix::GetStream());
      }
      return *handle_;
#endif
    }

  private:
    ~CublasHandler() {
      cublasDestroy(*handle_);
      if (handle_) {
        delete handle_;
      }
    }

#ifdef __APPLE__
    static boost::thread_specific_ptr<cublasHandle_t> handle_;
#else
    static thread_local cublasHandle_t* handle_;
#endif
};

typedef TMatrix<FVec> Matrix;
typedef TMatrix<IVec> IMatrix;

template <class M>
void Debug(const M& m, size_t pos = 0, size_t l = 5) {
  std::cerr << m.Rows() << " " << m.Cols() << std::endl;
  for(size_t i = 0; i < m.Rows(); ++i) {
    for(size_t j = pos; j < m.Cols() && j < pos + l; ++j) {
      std::cerr << m.GetVec()[i * m.Cols() + j] << " ";
    }
    std::cerr << std::endl;
    if(i == 4)
      break;
  }
}

Matrix& Swap(Matrix& Out, Matrix& In);

Matrix& Mean(Matrix& Out, const Matrix& In);

Matrix& Transpose(Matrix& Out, const Matrix& In);

Matrix& Transpose(Matrix& Out);

Matrix& Copy(Matrix& Out, const Matrix& In);

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r = 0,
                 const size_t c = 0);

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0,
                const size_t c = 0);

typedef std::pair<size_t, size_t> RowPair;
typedef std::vector<RowPair> RowPairs;
typedef thrust::device_vector<RowPair> DeviceRowPairs;

Matrix& Concat(Matrix& Out, const Matrix& In);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPair* devPairs,
                 size_t numPairs);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const RowPairs& pairs);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const thrust::device_vector<size_t>& indeces);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Softmax(Matrix& Out);

Matrix& LogSoftmax(Matrix& Out);

template <class Functor>
__global__ void gBroadcast(Functor functor,
                           float* out, const float* in1, const float* in2,
                           size_t rows, size_t rows1, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;

      const float* rowIn1 = in1 + (j % rows1) * cols;
      const float* rowIn2 = in2 + (j / rows1) * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowIn1[i], rowIn2[i]);
      }
    }
  }
}

template <class Functor>
Matrix& Broadcast(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows1 = Out.Rows();
  size_t rows2 = In.Rows();

  size_t rows = rows1 * rows2;
  size_t cols  = Out.Cols();

  Matrix Temp(rows, cols, 1.0);

  float* d_out = Temp.data();
  const float* d_in1 = Out.data();
  const float* d_in2 = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  gBroadcast<<<blocks, threads, 0, Matrix::GetStream()>>>(functor, d_out, d_in1, d_in2,
                                                          rows, rows1, cols);
  Swap(Out, Temp);
  return Out;
}

template <class Functor>
Matrix& BroadcastColumn(Functor functor, Matrix& Out, const Matrix& In) {
  // @TODO: Make this efficient with special kernel!
  Matrix InTemp;
  Transpose(InTemp, In);

  Transpose(Out);
  Broadcast(functor, Out, InTemp);
  Transpose(Out);
  return Out;
}

template <class Functor>
__global__ void gBroadcastVecColumn(Functor functor,
                                    float* out, const float* in, size_t rows, size_t cols) {
  extern __shared__ float sdata[];
  if (threadIdx.x < rows) {
      sdata[threadIdx.x] = in[threadIdx.x];
  }
  __syncthreads();

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    int index = noColumn;
    for (int noRow = 0; noRow < rows; ++noRow) {
        out[index] = functor(out[index], sdata[noRow]);
        index += cols;
    }
  }
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const thrust::device_vector<float>& In) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = thrust::raw_pointer_cast(In.data());

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);
  gBroadcastVecColumn<<<blocks, threads, rows * sizeof(float), Matrix::GetStream()>>>(functor, d_out, d_in, rows, cols);
  return Out;
}

// template <class Functor>
// __global__ void gBroadcastVec(Functor functor,
                              // float* out, const float* in, size_t rows, size_t cols) {
  // for(int bid = 0; bid < rows; bid += gridDim.x) {
    // int j = bid + blockIdx.x;
    // if(j < rows) {
      // float* rowOut = out + j * cols;
      // for(int tid = 0; tid < cols; tid += blockDim.x) {
        // int i = tid + threadIdx.x;
        // if(i < cols) {
          // rowOut[i] = functor(rowOut[i], in[i]);
        // }
      // }
    // }
  // }
// }

// template <class Functor>
// Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
  // size_t rows  = Out.Rows();
  // size_t cols = Out.Cols();

  // float* d_out = Out.data();
  // const float* d_in = In.data();

  // int blocks  = std::min(MAX_BLOCKS, (int)rows);
  // int threads = std::min(MAX_THREADS, (int)cols);
  // stream = Matrix::GetStream();
  // gBroadcastVec<<<blocks, threads, 0, stream>>>(functor, d_out, d_in, rows, cols);
  // return Out;
// }

template <class Functor>
__global__ void gBroadcastVec(Functor functor,
                              float* out, const float* in, size_t rows, size_t cols) {
  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    float vecValue = in[noColumn];
    int index = noColumn;
    for (int noRow = 0; noRow < rows; ++noRow) {
        out[index] = functor(out[index], vecValue);
        index += cols;
    }
  }
}

template <class Functor>
Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);
  stream = Matrix::GetStream();
  gBroadcastVec<<<blocks, threads, 0, stream>>>(functor, d_out, d_in, rows, cols);
  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor, float* out,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i]);;
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out, const float* in,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn = in + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn[i]);;
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out, const float* in1, const float* in2,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn1 = in1 + j * cols;
      const float* rowIn2 = in2 + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn1[i], rowIn2[i]);
      }
    }
  }
}

template <class Functor>
Matrix& Element(Functor functor, Matrix& Out) {
  float* d_out = Out.data();
  int blocks  = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  cudaStream_t& stream = Matrix::GetStream();
  gElement<<<blocks, threads, 0, stream>>>(functor, d_out, Out.Rows(), Out.Cols());
  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  cudaStream_t& stream = Matrix::GetStream();
  gElement<<<blocks, threads, 0, stream>>>(functor, d_out, d_in, Out.Rows(), Out.Cols());
  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In1, const Matrix& In2) {

  float* d_out = Out.data();
  const float* d_in1 = In1.data();
  const float* d_in2 = In2.data();

  int blocks  = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  cudaStream_t& stream = Matrix::GetStream();
  gElement<<<blocks, threads, 0, stream>>>(functor, d_out, d_in1, d_in2,
                                Out.Rows(), Out.Cols());
  return Out;
}

void SetColumn(Matrix& In, int noColumn, float value);
}  // namespace mblas
}  // namespace GPU
