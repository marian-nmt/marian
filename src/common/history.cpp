#include "history.h"
#include "sentence.h"

namespace amunmt {

Histories::Histories(const God &god, const Sentences& sentences)
 : coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    History *history = new History(god, sentence.GetLineNum());
    coll_[i].reset(history);
  }
}

///////////////////////////////////////////////////////////////////////////////
History::History(const  God &god, size_t lineNo)
: normalize_(god.Get<bool>("normalize"))
, lineNo_(lineNo)
{}


///////////////////////////////////////////////////////////////////////////////
class LineNumOrderer
{
public:
  bool operator()(const std::shared_ptr<History> &a, const std::shared_ptr<History> &b) const
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

