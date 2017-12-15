#include "histories.h"
#include "sentences.h"

using namespace std;

namespace amunmt {

Histories::Histories(const Sentences& sentences, bool normalizeScore)
 : coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = sentences.Get(i);
    History *history = new History(sentence, normalizeScore, 3 * sentence.size());
    coll_[i].reset(history);
  }
}

void Histories::Add(const Beams& beams) {
  for (size_t i = 0; i < size(); ++i) {
    if (!beams[i].empty()) {
      coll_[i]->Add(beams[i]);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
class LineNumOrderer
{
  public:
    bool operator()(const std::shared_ptr<History>& a, const std::shared_ptr<History>& b) const
    {
      return a->GetLineNum() < b->GetLineNum();
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////

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

Beam Histories::GetFirstHyps()
{
  Beam beam;
  for (auto& history : coll_) {
    beam.emplace_back(history->front()[0]);
  }
  return beam;
}

void Histories::Output(const God &god) const
{
  for (size_t i = 0; i < coll_.size(); ++i) {
    const History &history = *coll_.at(i);
    history.Output(god);

  }
}

}
