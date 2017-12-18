#pragma once

#include <map>
#include <numeric>
#include <boost/timer/timer.hpp>

#include "common/scorer.h"
#include "common/exception.h"
#include "common/god.h"
#include "common/utils.h"
#include "common/soft_alignment.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/nth_element.h"
#include "gpu/mblas/vector.h"
#include "gpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace GPU {

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const BestHyps &copy) = delete;
    BestHyps(const God &god);
    ~BestHyps();

    void DisAllowUNK(mblas::Matrix& Prob);

    // standard nth_element
    void FindBests(const BeamSize& beamSizes,
                   mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst);

    virtual void  CalcBeam(
        const Hypotheses& prevHyps,
        Scorer &scorer,
        const Words& filterIndices,
        std::vector<Hypotheses>& beams,
        BeamSize& beamSizes);

  private:
    std::unique_ptr<NthElement> nthElement_;
    mblas::Vector<unsigned> keys_;
    mblas::Vector<float> costs_;
    uint maxBeamSize_;

    // fast fused softmax and nth_element
    void FindBests(const BeamSize& beamSizes, mblas::Matrix& Probs,
    		mblas::Vector<NthOutBatch> &nBest,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst);

    void getNBestList(const BeamSize& beamSizes,
                      mblas::Matrix& Probs,
                      mblas::Vector<NthOutBatch> &nBest,
                      std::vector<float>& outCosts,
                      std::vector<uint>& outKeys,
                      const bool isFirst=false) const;

    void GetPairs(mblas::Vector<NthOutBatch> &nBest,
                  std::vector<uint>& outKeys,
                  std::vector<float>& outValues) const;


    std::vector<SoftAlignmentPtr> GetAlignments(const std::vector<ScorerPtr>& scorers,
                                                size_t hypIndex);

    std::vector<SoftAlignmentPtr> GetAlignments(Scorer &scorer,
                                                size_t hypIndex);

};

}
}

