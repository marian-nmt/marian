#pragma once
#include <memory>

#include "common/definitions.h"
#include "data/alignment.h"

namespace marian {

// one single (partial or full) hypothesis in beam search
// key elements:
//  - the word that this hyp ends with
//  - the aggregate score up to and including the word
//  - back pointer to previous hypothesis for traceback
class Hypothesis {
public:
  typedef IPtr<Hypothesis> PtrType;

private:
  // Constructors are private, use Hypothesis::New(...)

  Hypothesis() : prevHyp_(nullptr), prevBeamHypIdx_(0), word_(Word::ZERO), pathScore_(0.0) {}

  Hypothesis(const PtrType prevHyp,
             Word word,
             size_t prevBeamHypIdx, // beam-hyp index that this hypothesis originated from
             float pathScore)
      : prevHyp_(prevHyp), prevBeamHypIdx_(prevBeamHypIdx), word_(word), pathScore_(pathScore) {}

public:
 // Use this whenever creating a pointer to MemoryPiece
 template <class ...Args>
 static PtrType New(Args&& ...args) {
   return PtrType(new Hypothesis(std::forward<Args>(args)...));
 }

  const PtrType getPrevHyp() const { return prevHyp_; }

  Word getWord() const { return word_; }

  size_t getPrevStateIndex() const { return prevBeamHypIdx_; }

  float getPathScore() const { return pathScore_; }

  const std::vector<float>& getScoreBreakdown() { return scoreBreakdown_; }
  void setScoreBreakdown(const std::vector<float>& scoreBreaddown) { scoreBreakdown_ = scoreBreaddown; }

  const std::vector<float>& getAlignment() { return alignment_; }
  void setAlignment(const std::vector<float>& align) { alignment_ = align; };

  // helpers to trace back paths referenced from this hypothesis
  Words tracebackWords()
  {
      Words targetWords;
      for (auto hyp = this; hyp->getPrevHyp(); hyp = hyp->getPrevHyp().get()) {
        targetWords.push_back(hyp->getWord());
        // std::cerr << hyp->getWord() << " " << hyp << std::endl;
      }
      std::reverse(targetWords.begin(), targetWords.end());
      return targetWords;
  }

  // get soft alignments [t][s] -> P(s|t) for each target word starting from the hyp one
  typedef data::SoftAlignment SoftAlignment;
  SoftAlignment tracebackAlignment()
  {
      SoftAlignment align;
      for (auto hyp = this; hyp->getPrevHyp(); hyp = hyp->getPrevHyp().get()) {
          align.push_back(hyp->getAlignment());
      }
      std::reverse(align.begin(), align.end());
      return align; // [t][s] -> P(s|t)
  }

private:
  const PtrType prevHyp_;
  const size_t prevBeamHypIdx_;
  const Word word_;
  const float pathScore_;

  std::vector<float> scoreBreakdown_; // [num scorers]
  std::vector<float> alignment_;

  ENABLE_INTRUSIVE_PTR(Hypothesis)
};

typedef std::vector<IPtr<Hypothesis>> Beam;                // Beam = vector [beamSize] of hypotheses
typedef std::vector<Beam> Beams;                          // Beams = vector [batchDim] of vector [beamSize] of hypotheses
typedef std::tuple<Words, IPtr<Hypothesis>, float> Result; // (word ids for hyp, hyp, normalized sentence score for hyp)
typedef std::vector<Result> NBestList;                    // sorted vector of (word ids, hyp, sent score) tuples
}  // namespace marian
