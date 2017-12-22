#include <cassert>
#include <algorithm>
#include "sentences.h"

using namespace std;

namespace amunmt {

Sentences::Sentences()
  : maxLength_(0)
{}

Sentences::Sentences(size_t size)
:coll_(size)
,maxLength_(0)
{
}


Sentences::Sentences(const Sentences &other)
:coll_(other.coll_)
,maxLength_(other.maxLength_)
{
}

Sentences::~Sentences()
{}

const SentencePtr &Sentences::Get(size_t id) const
{
  return coll_.at(id);
}

void Sentences::Set(size_t id, const SentencePtr &sentence)
{
  assert(id < coll_.size());
  coll_[id] = sentence;
}

size_t Sentences::size() const {
  return coll_.size();
}

size_t Sentences::GetMaxLength() const {
  return maxLength_;
}

void Sentences::RecalcMaxLength()
{
  maxLength_ = 0;
  for (size_t i = 0; i < coll_.size(); ++i) {
    const SentencePtr &sentence = coll_[i];
    if (sentence && sentence->size() > maxLength_) {
      maxLength_ = sentence->size();
    }
  }
}

void Sentences::push_back(const SentencePtr &sentence) {
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
  //std::sort(coll_.begin(), coll_.end(), LengthOrderer());
  //std::random_shuffle ( coll_.begin(), coll_.end() );
}

SentencesPtr Sentences::NextMiniBatch(size_t batchsize, int batchWords)
{
  SentencesPtr sentences(new Sentences());

  if (batchWords) {
    size_t numWords = 0;
    size_t maxBatch = std::min(batchsize, size());
    //cerr << "maxBatch=" << maxBatch << endl;

    size_t ind = 0;
    while (ind < maxBatch) {
      SentencePtr sentence = coll_[ind];
      size_t sentLen = sentence->GetWords(0).size();

      if (sentences->size() && (numWords + sentLen) > batchWords) {
        // max batch
        break;
      }

      numWords += sentLen;

      // add next 32 sentences
      size_t endInd = std::min(size(), ind + 32);
      for (; ind < endInd; ++ind) {
        sentence = coll_[ind];
        sentences->push_back(sentence);

        if (ind == maxBatch) {
          break;
        }
      }
    }

    coll_.erase(coll_.begin(), coll_.begin() + ind);

    //cerr << "sentences=" << sentences->size() << " coll_=" << coll_.size() << endl;
  }
  else {
    size_t startInd = (batchsize > size()) ? 0 : size() - batchsize;
    for (size_t i = startInd; i < size(); ++i) {
      const SentencePtr &sentence = coll_[i];
      sentences->push_back(sentence);
    }

    coll_.resize(startInd);
  }

  return sentences;
}

}

