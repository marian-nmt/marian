#pragma once
#include <memory>
#include <cassert>
#include "common/types.h"
#include "common/soft_alignment.h"

namespace amunmt {

class Hypothesis;
class Sentence;

typedef std::shared_ptr<Hypothesis> HypothesisPtr;

class Hypothesis {
  public:
    Hypothesis(const Sentence &sentence)
    : sentence_(sentence),
       prevHyp_(nullptr),
       prevIndex_(0),
       word_(0),
       cost_(0.0)
    {}

    Hypothesis(const HypothesisPtr prevHyp, unsigned word, unsigned prevIndex, float cost)
    : sentence_(prevHyp->sentence_),
      prevHyp_(prevHyp),
      prevIndex_(prevIndex),
      word_(word),
      cost_(cost)
    {}

    Hypothesis(const HypothesisPtr prevHyp, unsigned word, unsigned prevIndex, float cost,
               std::vector<SoftAlignmentPtr> alignment)
    : sentence_(prevHyp->sentence_),
      prevHyp_(prevHyp),
      prevIndex_(prevIndex),
      word_(word),
      cost_(cost),
      alignments_(alignment)
    {}

    const HypothesisPtr GetPrevHyp() const {
      return prevHyp_;
    }

    unsigned GetWord() const {
      return word_;
    }

    unsigned GetPrevStateIndex() const {
      return prevIndex_;
    }

    float GetCost() const {
      return cost_;
    }

    std::vector<float>& GetCostBreakdown() {
      return costBreakdown_;
    }

    SoftAlignmentPtr GetAlignment(unsigned i) {
      assert(i < alignments_.size());
      return alignments_[i];
    }

    std::vector<SoftAlignmentPtr>& GetAlignments() {
      return alignments_;
    }

  private:
    const HypothesisPtr prevHyp_;
    const Sentence &sentence_;
    const unsigned prevIndex_;
    const unsigned word_;
    const float cost_;
    std::vector<SoftAlignmentPtr> alignments_;

    std::vector<float> costBreakdown_;
};

typedef std::pair<Words, HypothesisPtr> Result;
typedef std::vector<Result> NBestList;

////////////////////////////////////////////////////////////////////////////////////////////////////////


}

