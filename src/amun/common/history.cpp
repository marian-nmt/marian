#include "history.h"
#include "sentences.h"

using namespace std;

namespace amunmt {

History::History(const Sentence &sentence, bool normalizeScore, size_t maxLength)
  : normalize_(normalizeScore),
    lineNo_(sentence.GetLineNum()),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis(sentence))});
}

void History::Add(const Beam& beam) {
  if (beam.back()->GetPrevHyp() != nullptr) {
    for (size_t j = 0; j < beam.size(); ++j)
      if(beam[j]->GetWord() == EOS_ID || size() == maxLength_ ) {
        float cost = normalize_ ? beam[j]->GetCost() / history_.size() : beam[j]->GetCost();
        topHyps_.push({ history_.size(), j, cost });
      }
  }
  history_.push_back(beam);
}

NBestList History::NBest(size_t n) const
{
  NBestList nbest;
  auto topHypsCopy = topHyps_;
  while (nbest.size() < n && !topHypsCopy.empty()) {
    auto bestHypCoord = topHypsCopy.top();
    topHypsCopy.pop();

    size_t start = bestHypCoord.i;
    size_t j  = bestHypCoord.j;

    Words targetWords;
    HypothesisPtr bestHyp = history_[start][j];
    while(bestHyp->GetPrevHyp() != nullptr) {
      targetWords.push_back(bestHyp->GetWord());
      bestHyp = bestHyp->GetPrevHyp();
    }

    std::reverse(targetWords.begin(), targetWords.end());
    nbest.emplace_back(targetWords, history_[bestHypCoord.i][bestHypCoord.j]);
  }
  return nbest;
}


}

