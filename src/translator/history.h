#pragma once

#include "data/types.h"
#include "hypothesis.h"

#include <queue>

namespace marian {

// search grid of one batch entry
class History {
private:
  // one hypothesis of a full sentence (reference into search grid)
  struct SentenceHypothesisCoord {
    bool operator<(const SentenceHypothesisCoord& hc) const { return normalizedPathScore < hc.normalizedPathScore; }

    size_t timeStepIdx;        // last time step of this sentence hypothesis
    size_t beamIdx;            // which beam entry
    float normalizedPathScore; // length-normalized sentence score
  };

  float lengthPenalty(size_t length) { return std::pow((float)length, alpha_); }
  float wordPenalty(size_t length) { return wp_ * (float)length; }
public:
  History(size_t lineNo, float alpha = 1.f, float wp_ = 0.f);

  void add(const Beam& beam, Word trgEosId, bool last = false) {
    if(beam.back()->getPrevHyp() != nullptr) { // if not start hyp do
      for(size_t beamIdx = 0; beamIdx < beam.size(); ++beamIdx)
        if(beam[beamIdx]->getWord() == trgEosId || last) { // if this is a final hyp do
          float pathScore = (beam[beamIdx]->getPathScore() - wordPenalty(history_.size())) / lengthPenalty(history_.size()); // get and normalize path score
          topHyps_.push({history_.size(), beamIdx, pathScore}); // push final hyp on queue of scored hyps
        }
    }
    history_.push_back(beam);
  }

  size_t size() const { return history_.size(); } // number of time steps

  /* return n best hypotheses
   * @param n size of n-best list
   * @param skipEmpty skip empty hypotheses (see also: https://arxiv.org/abs/1908.10090)
   * @return at most max(n, beamSize) translation hypotheses
   * Note: if n is equal to the beam size, skipEmpty is true, and the empty hypothesis is in
   *       the top-n translations, the function will return less than n candidates. It is up to
   *       the caller to check the number of returned hypotheses.
   */
  NBestList nBest(size_t n, bool skipEmpty = false) const {
    NBestList nbest;
    for (auto topHypsCopy = topHyps_; nbest.size() < n && !topHypsCopy.empty(); topHypsCopy.pop()) {
      auto bestHypCoord = topHypsCopy.top();

      const size_t timeStepIdx = bestHypCoord.timeStepIdx; // last time step of this hypothesis
      const size_t beamIdx     = bestHypCoord.beamIdx;     // which beam entry
      Hypothesis::PtrType bestHyp = history_[timeStepIdx][beamIdx];

      // trace back best path
      Words targetWords = bestHyp->tracebackWords();
      if (skipEmpty && targetWords.size() == 0)
        continue; // skip empty translation
      // note: bestHyp->getPathScore() is not normalized, while bestHypCoord.normalizedPathScore is
      nbest.emplace_back(targetWords, bestHyp, bestHypCoord.normalizedPathScore);
    }
    return nbest;
  }

  Result top() const {
    const NBestList& nbest = nBest(1);
    ABORT_IF(nbest.empty(), "No hypotheses in n-best list??");
    return nbest[0];
  }

  size_t getLineNum() const { return lineNo_; }

private:
  std::vector<Beam> history_; // [time step][index into beam] search grid @TODO: simplify as this is currently an expensive length count
  std::priority_queue<SentenceHypothesisCoord> topHyps_; // all sentence hypotheses (those that reached eos), sorted by score
  size_t lineNo_;
  float alpha_;
  float wp_;
};

typedef std::vector<Ptr<History>> Histories; // [batchDim]
}  // namespace marian
