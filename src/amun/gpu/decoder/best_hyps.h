#pragma once

#include <map>
#include <numeric>
#include <boost/timer/timer.hpp>

#include "common/scorer.h"
#include "common/exception.h"
#include "common/god.h"
#include "common/utils.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/nth_element.h"
#include "gpu/mblas/vector.h"

#include "gpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace GPU {

class BestHyps : public BaseBestHyps
{
  public:
    BestHyps(const BestHyps &copy) = delete;
    BestHyps(const God &god);

    void DisAllowUNK(mblas::Matrix& Prob);

    // standard nth_element
    void FindBests(const std::vector<unsigned>& beamSizes, mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst);

    std::vector<SoftAlignmentPtr> GetAlignments(const std::vector<ScorerPtr>& scorers,
                                                unsigned hypIndex);

    void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<unsigned>& beamSizes);

  private:
    std::unique_ptr<NthElement> nthElement_;
    mblas::Vector<unsigned> keys_;
    mblas::Vector<float> costs_;
    unsigned maxBeamSize_;

    // fast fused softmax and nth_element
    void FindBests(const std::vector<unsigned>& beamSizes, mblas::Matrix& Probs,
    		mblas::Vector<NthOutBatch> &nBest,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst);

    void getNBestList(const std::vector<unsigned>& beamSizes,
                      mblas::Matrix& Probs,
                      mblas::Vector<NthOutBatch> &nBest,
                      std::vector<float>& outCosts,
                      std::vector<unsigned>& outKeys,
                      const bool isFirst=false) const;

    void GetPairs(mblas::Vector<NthOutBatch> &nBest,
                  std::vector<unsigned>& outKeys,
                  std::vector<float>& outValues) const;

};

}
}

