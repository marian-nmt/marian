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

namespace lib = thrust;
namespace iterlib = thrust;

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
      strm << Rows() << "x" << Cols() << ":"; // ":\n";
      for (size_t row = 0; row < Rows(); ++row) {
        float rowSum = 0;
        for (size_t col = 0; col < Cols(); ++col) {
          //strm << (*this)(row, col) << " ";
          rowSum += (*this)(row, col);
        }
        //strm << std::endl;
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

    virtual void BestHyps(Beam& bestHyps,
      const Beam& prevHyps,
      BaseMatrices& ProbsEnsemble,
      const size_t beamSize,
      History& history,
			const std::vector<ScorerPtr> &scorers,
			const Words &filterIndices,
      bool returnAlignment) const
    {
	  using namespace mblas;
	  typedef TMatrix<VecType> M;

	  auto& weights = God::GetScorerWeights();

	  M& Probs = static_cast<M&>(*ProbsEnsemble[0]);

	  M Costs(Probs.Rows(), 1);
	  HostVector<float> vCosts;
	  for(auto& h : prevHyps)
		vCosts.push_back(h->GetCost());
	  algo::copy(vCosts.begin(), vCosts.end(), Costs.begin());

	  BroadcastVecColumn(weights[scorers[0]->GetName()] * _1 + _2,
						 Probs, Costs);
	  for(size_t i = 1; i < ProbsEnsemble.size(); ++i) {
		  M &currProbs = static_cast<M&>(*ProbsEnsemble[i]);

		  Element(_1 + weights[scorers[i]->GetName()] * _2,
				Probs, currProbs);
	  }

	  DeviceVector<unsigned> keys(Probs.size());
	  HostVector<unsigned> bestKeys(beamSize);
	  HostVector<float> bestCosts(beamSize);

	  // @TODO: make this more efficient
	  if (!God::Get<bool>("allow-unk")) {
        for(size_t i = 0; i < Probs.Rows(); i++)
            Probs.Set(i, UNK, std::numeric_limits<float>::lowest());
        }

        /*
        thrust::sequence(keys.begin(), keys.end());
        thrust::nth_element(keys.begin(), keys.begin() + beamSize, keys.end(),
                            ProbCompare(Probs.data()));

        for(int i = 0; i < beamSize; ++i) {
            bestKeys[i] = keys[i];
            // solve this better
            bestCosts[i] = Probs.GetVec()[keys[i]];
        }*/

        // @TODO: Here we need to have a partial sort
        if (beamSize < 10) {
        for (size_t i = 0; i < beamSize; ++i) {
            DeviceVector<float>::iterator iter =
            algo::max_element(Probs.begin(), Probs.end());
            bestKeys[i] = iter - Probs.begin();
            bestCosts[i] = *iter;
            *iter = std::numeric_limits<float>::lowest();
        }
        algo::copy(bestKeys.begin(), bestKeys.end(), keys.begin());
	  }
	  else {
        // these two function do not have equivalents in
        // in the standard library or boost, keeping thrust
        // namespace for now
        thrust::sequence(keys.begin(), keys.end());
        thrust::sort_by_key(Probs.begin(), Probs.end(),
                            keys.begin(), algo::greater<float>());

        algo::copy_n(keys.begin(), beamSize, bestKeys.begin());
        algo::copy_n(Probs.begin(), beamSize, bestCosts.begin());
	  }


	  std::vector<HostVector<float>> breakDowns;
	  bool doBreakdown = God::Get<bool>("n-best");
	  if (doBreakdown) {
        breakDowns.push_back(bestCosts);
        for (size_t i = 1; i < ProbsEnsemble.size(); ++i) {
            HostVector<float> modelCosts(beamSize);
            M &currProbs = static_cast<M&>(*ProbsEnsemble[i]);

            auto it = iteralgo::make_permutation_iterator(currProbs.begin(), keys.begin());
            algo::copy(it, it + beamSize, modelCosts.begin());
            breakDowns.push_back(modelCosts);
        }
	  }


    bool filter = God::Get<std::vector<std::string>>("softmax-filter").size();

    for (size_t i = 0; i < beamSize; i++) {
    size_t wordIndex = bestKeys[i] % Probs.Cols();
    if (filter) {
      wordIndex = filterIndices[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / Probs.Cols();
    float cost = bestCosts[i];

    HypothesisPtr hyp;
    if (returnAlignment) {
      std::vector<SoftAlignmentPtr> alignments;
      for (auto& scorer : scorers) {
        if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
          auto& attention = encdec->GetAttention();
          size_t attLength = attention.Cols();

          alignments.emplace_back(new SoftAlignment(attention.begin() + hypIndex * attLength,
                                                    attention.begin() + (hypIndex + 1) * attLength));
      } else {
        UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
      }
    }
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost, alignments));
    } else {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
    }

    if(doBreakdown) {
    hyp->GetCostBreakdown().resize(ProbsEnsemble.size());
    float sum = 0;
    for (size_t j = 0; j < ProbsEnsemble.size(); ++j) {
        if (j == 0)
        hyp->GetCostBreakdown()[0] = breakDowns[0][i];
        else {
        float cost = 0;
        if (j < ProbsEnsemble.size()) {
            if(prevHyps[hypIndex]->GetCostBreakdown().size() < ProbsEnsemble.size())
            const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(ProbsEnsemble.size(), 0.0);
            cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
        }
        sum += weights[scorers[j]->GetName()] * cost;
        hyp->GetCostBreakdown()[j] = cost;
        }
    }
    hyp->GetCostBreakdown()[0] -= sum;
    hyp->GetCostBreakdown()[0] /= weights[scorers[0]->GetName()];
    }
    bestHyps.push_back(hyp);
    }
  }

  private:
    size_t rows_;
    size_t cols_;
    VecType data_;
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
    }
    return *handle_;
#endif
  }

private:
  ~CublasHandler()
  {
	// not called. Leaking handles
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
                 const size_t r = 0, const size_t c = 0);

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0, const size_t c = 0);

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
                 const std::vector<size_t>& indeces);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Softmax(Matrix& Out);

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
Matrix& Broadcast(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
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
  gBroadcast<<<blocks, threads, 0, stream>>>(functor, d_out, d_in1, d_in2,
                                             rows, rows1, cols);
  cudaStreamSynchronize(stream);
  Swap(Out, Temp);
  return Out;
}

template <class Functor>
Matrix& BroadcastColumn(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
  // @TODO: Make this efficient with special kernel!
  Matrix InTemp;
  Transpose(InTemp, In);

  Transpose(Out);
  Broadcast(functor, Out, InTemp, stream);
  Transpose(Out);
  return Out;
}

template <class Functor>
__global__ void gBroadcastVecColumn(Functor functor,
                                    float* out, const float* in, size_t rows, size_t cols) {
  for(int bid = 0; bid < cols; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < cols) {
      for(int tid = 0; tid < rows; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < rows) {
          float* rowOut = out + i * cols + j;
          const float* rowIn  = in + i;
          *rowOut = functor(*rowOut, *rowIn);
        }
      }
    }
  }
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)cols);
  int threads = std::min(MAX_THREADS, (int)rows);
  gBroadcastVecColumn<<<blocks, threads, 0, stream>>>(functor, d_out, d_in, rows, cols);
  cudaStreamSynchronize(stream);
  return Out;
}

template <class Functor>
__global__ void gBroadcastVec(Functor functor,
                              float* out, const float* in, size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          rowOut[i] = functor(rowOut[i], in[i]);
        }
      }
    }
  }
}

template <class Functor>
Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
  //Broadcast(functor, Out, In, stream);
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  gBroadcastVec<<<blocks, threads, 0, stream>>>(functor, d_out, d_in, rows, cols);
  cudaStreamSynchronize(stream);
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
  gElement<<<blocks, threads>>>(functor, d_out, Out.Rows(), Out.Cols());
  cudaStreamSynchronize(0);
  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  gElement<<<blocks, threads>>>(functor, d_out, d_in, Out.Rows(), Out.Cols());
  cudaStreamSynchronize(0);
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
  gElement<<<blocks, threads>>>(functor, d_out, d_in1, d_in2,
                                Out.Rows(), Out.Cols());
  cudaStreamSynchronize(0);
  return Out;
}

}
}
