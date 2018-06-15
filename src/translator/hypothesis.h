#pragma once
#include <memory>

#include "common/definitions.h"

namespace marian {

class Hypothesis {
public:
  Hypothesis() : prevHyp_(nullptr), prevIndex_(0), word_(0), cost_(0.0) {}

  Hypothesis(const Ptr<Hypothesis> prevHyp,
             size_t word,
             size_t prevIndex,
             float cost)
      : prevHyp_(prevHyp), prevIndex_(prevIndex), word_(word), cost_(cost) {}

  const Ptr<Hypothesis> GetPrevHyp() const { return prevHyp_; }

  size_t GetWord() const { return word_; }

  size_t GetPrevStateIndex() const { return prevIndex_; }

  float GetCost() const { return cost_; }

  std::vector<float>& GetCostBreakdown() { return costBreakdown_; }
  std::vector<float>& GetAlignment() { return alignment_; }

  void SetAlignment(const std::vector<float>& align) { alignment_ = align; };

private:
  const Ptr<Hypothesis> prevHyp_;
  const size_t prevIndex_;
  const size_t word_;
  const float cost_;

  std::vector<float> costBreakdown_;
  std::vector<float> alignment_;
};

typedef std::vector<Ptr<Hypothesis>> Beam;
typedef std::vector<Beam> Beams;
typedef std::vector<size_t> Words;
typedef std::tuple<Words, Ptr<Hypothesis>, float> Result;
typedef std::vector<Result> NBestList;
}
