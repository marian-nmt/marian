#pragma once

#include "types.h"

class Hypothesis {
 public:
    Hypothesis(size_t word, size_t prev, float cost)
      : prev_(prev),
        word_(word),
        cost_(cost) {
    }

    size_t GetWord() const {
      return word_;
    }

    size_t GetPrevStateIndex() const {
      return prev_;
    }

    float GetCost() const {
      return cost_;
    }
    
    std::vector<float>& GetCostBreakdown() {
      return costBreakdown_;
    }

 private:
    const size_t prev_;
    const size_t word_;
    const float cost_;
    
    std::vector<float> costBreakdown_;
};

typedef std::vector<Hypothesis> Beam;
typedef std::pair<Sentence, Hypothesis> Result;
typedef std::vector<Result> NBestList;
