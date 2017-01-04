#include "history.h"
#include "sentence.h"

Histories::Histories(const Sentences& sentences)
:coll_(sentences.size())
{
  for (size_t i = 0; i < sentences.size(); ++i) {
    const Sentence &sentence = *sentences.at(i).get();
    History &history = coll_[i];
    history.SetLineNum(sentence.GetLineNum());
  }
}

