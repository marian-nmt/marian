#include <algorithm>
#include "sentences.h"
#include "god.h"
#include "translation_task.h"

namespace amunmt {

Sentences::Sentences()
  : maxLength_(0)
{}

Sentences::~Sentences()
{}

void Sentences::push_back(SentencePtr sentence) {
  const Words &words = sentence->GetWords(0);
  size_t len = words.size();
  if (len > maxLength_) {
    maxLength_ = len;
  }

  coll_.push_back(sentence);
}

class LengthOrderer {
 public:
  bool operator()(const SentencePtr& a, const SentencePtr& b) const {
    return a->GetWords(0).size() < b->GetWords(0).size();
  }
};

void Sentences::SortByLength() {
  std::sort(coll_.rbegin(), coll_.rend(), LengthOrderer());
}

SentencesPtr Sentences::NextMiniBatch(size_t batchsize)
{
  SentencesPtr sentences(new Sentences());
  size_t startInd = (batchsize > size()) ? 0 : size() - batchsize;
  for (size_t i = startInd; i < size(); ++i) {
    SentencePtr sentence = at(i);
    sentences->push_back(sentence);
  }

  coll_.resize(startInd);
  return sentences;
}

}

