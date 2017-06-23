#include "history.h"

#include "sentences.h"

namespace amunmt {

History::History(size_t lineNo, bool normalizeScore, size_t maxLength)
  : normalize_(normalizeScore),
    lineNo_(lineNo),
   maxLength_(maxLength)
{
  Add({HypothesisPtr(new Hypothesis())});
}


Histories::Histories(const Sentences& sentences, bool normalizeScore)
 : coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    History *history = new History(sentence.GetLineNum(), normalizeScore, 3 * sentence.size());
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
  for (size_t i = 0; i < other.size(); ++i) {
    std::shared_ptr<History> history = other.coll_[i];
    coll_.push_back(history);
  }
}

}

