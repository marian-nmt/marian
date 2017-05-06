#include "history.h"
#include "sentence.h"
#include "sentences.h"

namespace amunmt {

History::History(size_t lineNo, bool normalize)
  : normalize_(normalize),
    lineNo_(lineNo)
{}


void History::Add(const Beam& beam, bool last)
{
  if (beam.back()->GetPrevHyp() != nullptr) {
    for (size_t j = 0; j < beam.size(); ++j)
      if(beam[j]->GetWord() == EOS_ID || last) {
        float cost = normalize_ ? beam[j]->GetCost() / history_.size() : beam[j]->GetCost();
        topHyps_.push({ history_.size(), j, cost });
      }
  }
  history_.push_back(beam);
}


size_t History::size() const
{
  return history_.size();
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


Result History::Top() const
{
  return NBest(1)[0];
}


size_t History::GetLineNum() const
{
  return lineNo_;
}

Histories::Histories(const Sentences& sentences, bool normalize)
{
  for (const auto& sentence : sentences) {
    coll_.emplace_back(new History(sentence->GetLineNum(), normalize));
    std::cerr <<  "SIZE: " << size() << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////////


class LineNumOrderer
{
  public:
    bool operator()(const std::shared_ptr<History>& a, const std::shared_ptr<History>& b) const
    {
      return a->GetLineNum() < b->GetLineNum();
    }
};

void Histories::SortByLineNum()
{
  std::sort(coll_.begin(), coll_.end(), LineNumOrderer());
}

void Histories::Append(const Histories &other)
{
  for (size_t i = 0; i < other.size(); ++i) {
    std::shared_ptr<History> history = other.coll_[i];
    coll_.push_back(history);
  }
}

std::shared_ptr<History> Histories::at(size_t id) const {
  return coll_.at(id);
}

size_t Histories::size() const {
  return coll_.size();
}

Histories::Histories()
{}

}

