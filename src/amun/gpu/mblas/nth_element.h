#pragma once

#include <vector>
#include <algorithm>

#include <cuda.h>
#include "gpu/mblas/matrix.h"

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

inline std::ostream& operator<<(std::ostream &out, const NthOut &obj)
{
  out << "(" << obj.ind << "," << obj.score << ")";
  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////

class NthElement {
  public:
    NthElement() = delete;
    NthElement(const NthElement &copy) = delete;
    NthElement(uint maxBeamSize, uint maxBatchSize);
    virtual ~NthElement();

    void getNBestList(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                      std::vector<float>& outCosts, std::vector<uint>& outKeys,
                      const bool isFirst=false);

    void GetPairs(uint number,
                  std::vector<uint>& outKeys,
                  std::vector<float>& outValues);

    void getValueByKey(std::vector<float>& out, const mblas::Matrix &d_in) const;

  private:
    const uint BLOCK_SIZE = 512;

    mblas::TMatrix<NthOut> d_out;

    mblas::TMatrix<NthOut> d_res;
    //HostVector<NthOut> h_res;
    std::vector<NthOut> h_res;

    mblas::TMatrix<float> d_breakdown;
    mblas::TMatrix<uint> d_batchPosition;
    mblas::TMatrix<uint> d_cumBeamSizes;

    uint maxBeamSize_, maxBatchSize_;

    void getNBestList(mblas::Matrix &probs,
                      const HostVector<uint>& batchFirstElementIdxs,
                      const HostVector<uint>& cummulatedBeamSizes);

};

}  // namespace GPU
}
