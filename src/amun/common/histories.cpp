#include "histories.h"
#include "sentences.h"

using namespace std;

namespace amunmt {

Histories::Histories(const Sentences& sentences, bool normalizeScore, float maxLengthMult)
 : coll_(sentences.size())
{
  for (unsigned i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = sentences.Get(i);
    History *history = new History(sentence, normalizeScore, maxLengthMult * (float) sentence.size());
    coll_[i].reset(history);
  }
}


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
  for (unsigned i = 0; i < other.size(); ++i) {
    std::shared_ptr<History> history = other.coll_[i];
    coll_.push_back(history);
  }
}

void Histories::SetActive(bool active)
{
  for (size_t i = 0; i < coll_.size(); ++i) {
    SetActive(i, false);
  }
}

void Histories::SetActive(unsigned id, bool active)
{
  coll_[id]->SetActive(active);
}

unsigned Histories::NumActive() const
{
  unsigned ret = 0;
  for (size_t i = 0; i < coll_.size(); ++i) {
    if (coll_[i]->GetActive()) {
      ++ret;
    }
  }
  return ret;
}


}
