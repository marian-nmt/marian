#pragma once

#include "tensor_wrapper.h"
#include "vector_wrapper.h"

namespace amunmt {
namespace GPU {

struct NthOut
{
  unsigned ind;
  float score;

  __device__ __host__
  NthOut() {}

  __device__ __host__
  NthOut(unsigned init)
  :ind(init)
  ,score(init)
  {}

  __device__ __host__
  NthOut(unsigned &vInd, float vScore)
  :ind(vInd)
  ,score(vScore)
  {}

  __device__ __host__
  NthOut& operator+=(const NthOut& rhs)
  {
    ind += rhs.ind;
    score += rhs.score;
    return *this;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////

struct NthOutBatch
{
  unsigned ind;
  float score;
  //unsigned hypoInd;
  //unsigned vocabInd;

  __device__ __host__
  NthOutBatch(const float& rhs)
  {
    // only to be used to init variable in tensor.h gSum
    assert(rhs == 0.0f);
    ind = rhs;
    score = rhs;
    //hypoInd = rhs;
    //vocabInd = rhs;
  }

  __device__ __host__
  NthOutBatch() {}

  __device__ __host__
  NthOutBatch(unsigned vInd, float vScore, unsigned vHypoInd, unsigned vVocabInd)
  :ind(vInd)
  ,score(vScore)
  //,hypoInd(vHypoInd)
  //,vocabInd(vVocabInd)
  {}

  __device__ __host__
  NthOutBatch& operator=(const NthOutBatch& rhs)
  {
    ind = rhs.ind;
    score = rhs.score;
    //hypoInd = rhs.hypoInd;
    //vocabInd = rhs.vocabInd;
    return *this;
  }

  __device__ __host__
  NthOutBatch& operator+=(const NthOutBatch& rhs)
  {
    ind += rhs.ind;
    score += rhs.score;
    //hypoInd += rhs.hypoInd;
    //vocabInd += rhs.vocabInd;
    return *this;
  }

};


/////////////////////////////////////////////////////////////////////////////////////////

inline std::ostream& operator<<(std::ostream &out, const NthOut &obj)
{
  out << "(" << obj.ind << "," << obj.score << ")";
  return out;
}

inline std::ostream& operator<<(std::ostream &out, const NthOutBatch &obj)
{
  out << "("
      << obj.ind << ","
      << obj.score << ","
      //<< obj.hypoInd << ","
      //<< obj.vocabInd
      << ")";
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////

__global__ void gMaxElement(mblas::VectorWrapper<NthOut> out,
                            const mblas::TensorWrapper<float> probsWrap,
                            const mblas::VectorWrapper<unsigned> batchPositionWrap,
                            unsigned numBatches);

__global__ void gMaxElementUpdate(mblas::VectorWrapper<NthOut> out,
                                  mblas::TensorWrapper<float> probsWrap,
                                  mblas::VectorWrapper<NthOut> resWrap,
                                  const mblas::VectorWrapper<unsigned> batchPositionWrap,
                                  const mblas::VectorWrapper<unsigned> cumBeamSizesWrap,
                                  unsigned numBlocks);

__global__ void gGetValueByKey(mblas::TensorWrapper<float> out,
                              const   mblas::TensorWrapper<float> in,
                              unsigned* indices, unsigned n);

}
}
