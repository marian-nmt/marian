#include "history.h"
#include "sentence.h"

Histories::Histories(const Sentences& sentences)
:coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    History *history = new History(sentence.GetLineNum());
    coll_[i].reset(history);
  }
}

class LineNumOrderer
{
public:
  bool operator()(const boost::shared_ptr<History> &a, const boost::shared_ptr<History> &b) const
  {
    return a->GetLineNum() < b->GetLineNum();
  }

};

void Histories::SortByLineNum()
{
  std::sort(coll_.begin(), coll_.end(), LineNumOrderer());
}
