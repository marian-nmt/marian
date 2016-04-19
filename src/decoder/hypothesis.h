#pragma once

#include "types.h"

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
    
 private:
    const HypothesisPtr prevHyp_;
    const size_t prevIndex_;
    const size_t word_;
    const float cost_;
    
    std::vector<float> costBreakdown_;
};

typedef std::vector<HypothesisPtr> Beam;
typedef std::pair<Sentence, HypothesisPtr> Result;
typedef std::vector<Result> NBestList;
