#pragma once

#include "matrix_wrapper.h"

namespace amunmt {
namespace GPU {

struct NthOut
{
  uint ind;
  float score;

  __device__ __host__
  NthOut() {}

  __device__ __host__
  NthOut(uint init)
  :ind(init)
  ,score(init)
  {}

  __device__ __host__
  NthOut(uint &vInd, float vScore)
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
  uint ind;
  float score;
  //uint hypoInd;
  //uint vocabInd;

  __device__ __host__
  NthOutBatch(const float& rhs)
  {
    // only to be used to init variable in matrix.h gSum
    assert(rhs == 0.0f);
    ind = rhs;
    score = rhs;
    //hypoInd = rhs;
    //vocabInd = rhs;
  }

  __device__ __host__
  NthOutBatch() {}

  __device__ __host__
  NthOutBatch(uint vInd, float vScore, uint vHypoInd, uint vVocabInd)
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

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut> out,
                            const mblas::MatrixWrapper<float> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches);

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut> out,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<NthOut> resNewWrap,
                                  const mblas::MatrixWrapper<uint> batchPositionWrap,
                                  const mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks);

__global__ void gGetValueByKey(mblas::MatrixWrapper<float> out,
                              const   mblas::MatrixWrapper<float> in,
                              uint* indices, uint n);

}
}
