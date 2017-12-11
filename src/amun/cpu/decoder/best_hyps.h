#pragma once

#include <vector>
#include <boost/iterator/permutation_iterator.hpp>

#include "common/scorer.h"
#include "common/god.h"
#include "common/exception.h"
#include "cpu/mblas/matrix.h"
#include "cpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace CPU {

struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const God &god)
      : BestHypsBase(god)
    {}

    virtual void  CalcBeam(
                          const Beam& prevHyps,
                          Scorer &scorer,
                          const Words& filterIndices,
                          std::vector<Beam>& beams,
                          std::vector<uint>& beamSizes)
    {
      assert(false);
    }

};

}  // namespace CPU
}  // namespace amunmt
