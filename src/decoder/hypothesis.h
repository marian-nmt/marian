#pragma once
#include <iostream>
#include <tuple>
#include "types.h"

class Hypothesis;

//typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef Hypothesis* HypothesisPtr;

class Hypothesis {
  friend std::ostream& operator<<(std::ostream &out, const Hypothesis &obj);

 public:
    Hypothesis(const std::tuple<>&)
     : prevHyp_(nullptr),
       prevIndex_(0),
       word_(0),
       cost_(0.0)
    {}
    
    Hypothesis(const std::tuple<HypothesisPtr, size_t, size_t, float>& tuple)
      : prevHyp_(std::get<0>(tuple)),
        word_(std::get<1>(tuple)),
        prevIndex_(std::get<2>(tuple)),
        cost_(std::get<3>(tuple))
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
typedef std::pair<Words, HypothesisPtr> Result;
typedef std::vector<Result> NBestList;

