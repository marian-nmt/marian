#include "history.h"
#include "sentences.h"

using namespace std;

namespace amunmt {

History::History(const Sentence &sentence, bool normalizeScore, unsigned maxLength)
  : normalize_(normalizeScore),
    lineNo_(sentence.GetLineNum()),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis(sentence))});
}

void History::Add(const Beam& beam) {
  if (beam.back()->GetPrevHyp() != nullptr) {
    for (unsigned j = 0; j < beam.size(); ++j)
      if(beam[j]->GetWord() == EOS_ID || size() == maxLength_ ) {
        float cost = normalize_ ? beam[j]->GetCost() / history_.size() : beam[j]->GetCost();
        topHyps_.push({ (unsigned)history_.size(), j, cost });
      }
  }
  history_.push_back(beam);
}

NBestList History::NBest(unsigned n) const
{
  NBestList nbest;
  auto topHypsCopy = topHyps_;
  while (nbest.size() < n && !topHypsCopy.empty()) {
    auto bestHypCoord = topHypsCopy.top();
    topHypsCopy.pop();

    unsigned start = bestHypCoord.i;
    unsigned j  = bestHypCoord.j;

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

void History::SetActive(bool active)
{
  active_ = active;
}

bool History::GetActive() const
{
  return active_;
}

}

