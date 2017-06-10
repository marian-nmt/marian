#pragma once

#include <vector>
#include <algorithm>

#include <cuda.h>
#include "gpu/mblas/matrix.h"

namespace amunmt {
namespace GPU {

struct NthOut
{
  int ind;
  float score;

  NthOut() {}

  NthOut(int val)
  :ind(val)
  ,score(val)
  {}

  __device__ __host__
  NthOut(int &vInd, float vScore)
  :ind(vInd)
  ,score(vScore)
  {}

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
    NthElement(size_t maxBeamSize, size_t maxBatchSize, cudaStream_t& stream);
    virtual ~NthElement();

    void getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                      std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                      const bool isFirst=false);

    void GetPairs(size_t number,
                  std::vector<unsigned>& outKeys,
                  std::vector<float>& outValues);

    void getValueByKey(std::vector<float>& out, const mblas::Matrix &d_in) const;

  private:
    const int BLOCK_SIZE = 512;
    const int numBlocks_;
    cudaStream_t& stream_;

    DeviceVector<NthOut> d_out;

    DeviceVector<NthOut> d_res;

    HostVector<NthOut> h_res;

    DeviceVector<float> d_breakdown;
    DeviceVector<int> d_batchPosition;
    DeviceVector<int> d_cumBeamSizes;

    size_t maxBeamSize_, maxBatchSize_;

    void getNBestList(mblas::Matrix &probs, const std::vector<int>& batchFirstElementIdxs,
                              const std::vector<int>& cummulatedBeamSizes);

};

}  // namespace GPU
}
