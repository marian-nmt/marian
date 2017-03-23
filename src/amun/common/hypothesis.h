#pragma once
#include <memory>
#include "common/types.h"
#include "common/soft_alignment.h"

namespace amunmt {

class Hypothesis;

typedef std::shared_ptr<Hypothesis> HypothesisPtr;

class Hypothesis {
  public:
    Hypothesis()
     : prevHyp_(nullptr),
       prevIndex_(0),
       word_(0),
       cost_(0.0)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost)
      : prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        cost_(cost)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost,
               std::vector<SoftAlignmentPtr> alignment)
      : prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        cost_(cost),
        alignments_(alignment)
    {}

    const HypothesisPtr GetPrevHyp() const {
      return prevHyp_;
    }

    size_t GetWord() const {
      return word_;
    }

    size_t GetPrevStateIndex() const {
      return prevIndex_;
    }

    float GetCost() const {
      return cost_;
    }

    std::vector<float>& GetCostBreakdown() {
      return costBreakdown_;
    }

    SoftAlignmentPtr GetAlignment(size_t i) {
      return alignments_[i];
    }

    std::vector<SoftAlignmentPtr>& GetAlignments() {
      return alignments_;
    }

  private:
    const HypothesisPtr prevHyp_;
    const size_t prevIndex_;
    const size_t word_;
    const float cost_;
    std::vector<SoftAlignmentPtr> alignments_;

    std::vector<float> costBreakdown_;
};

typedef std::vector<HypothesisPtr> Beam;
typedef std::vector<Beam> Beams;
typedef std::pair<Words, HypothesisPtr> Result;
typedef std::vector<Result> NBestList;

}

