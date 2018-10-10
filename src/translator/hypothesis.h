#pragma once
#include <memory>

#include "common/definitions.h"
#include "data/alignment.h"

namespace marian {

class Hypothesis {
public:
  Hypothesis() : prevHyp_(nullptr), prevIndex_(0), word_(0), pathScore_(0.0) {}

  Hypothesis(const Ptr<Hypothesis> prevHyp,
             Word word,
             IndexType prevIndex,
             float pathScore)
      : prevHyp_(prevHyp), prevIndex_(prevIndex), word_(word), pathScore_(pathScore) {}

  const Ptr<Hypothesis> GetPrevHyp() const { return prevHyp_; }

  Word GetWord() const { return word_; }

  IndexType GetPrevStateIndex() const { return prevIndex_; }

  float GetPathScore() const { return pathScore_; }

  std::vector<float>& GetScoreBreakdown() { return scoreBreakdown_; }
  std::vector<float>& GetAlignment() { return alignment_; }

  void SetAlignment(const std::vector<float>& align) { alignment_ = align; };

  // helpers to trace back paths referenced from this hypothesis
  Words TracebackWords()
  {
      Words targetWords;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          targetWords.push_back(hyp->GetWord());
          // std::cerr << hyp->GetWord() << " " << hyp << std::endl;
      }
      std::reverse(targetWords.begin(), targetWords.end());
      return targetWords;
  }

  // get soft alignments for each target word starting from the hyp one
  typedef data::SoftAlignment SoftAlignment;
  SoftAlignment TracebackAlignment()
  {
      SoftAlignment align;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          align.push_back(hyp->GetAlignment());
      }
      std::reverse(align.begin(), align.end());
      return align;
  }

private:
  const Ptr<Hypothesis> prevHyp_;
  const IndexType prevIndex_;
  const Word word_;
  const float pathScore_;

  std::vector<float> scoreBreakdown_;
  std::vector<float> alignment_;
};

typedef std::vector<Ptr<Hypothesis>> Beam;                // Beam = vector of hypotheses
typedef std::vector<Beam> Beams;                          // Beams = vector of vector of hypotheses
typedef std::tuple<Words, Ptr<Hypothesis>, float> Result; // (word ids for hyp, hyp, normalized sentence score for hyp)
typedef std::vector<Result> NBestList;                    // sorted vector of (word ids, hyp, sent score) tuples
}  // namespace marian
