#include <algorithm>
#include <sstream>
#include "sentences.h"

using namespace std;

namespace amunmt {

Sentences::Sentences()
  : maxLength_(0)
{}

Sentences::~Sentences()
{}

SentencePtr Sentences::at(unsigned id) const
{
  return coll_.at(id);
}

const Sentence &Sentences::Get(unsigned id) const
{
  return *coll_.at(id);
}

unsigned Sentences::size() const {
  return coll_.size();
}

unsigned Sentences::GetMaxLength() const {
  return maxLength_;
}

void Sentences::push_back(SentencePtr sentence) {
  const Words &words = sentence->GetWords(0);
  unsigned len = words.size();
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

SentencesPtr Sentences::NextMiniBatch(unsigned batchsize, int batchWords)
{
  SentencesPtr sentences(new Sentences());

  if (batchWords) {
    unsigned numWords = 0;
    unsigned maxBatch = std::min(batchsize, size());
    //cerr << "maxBatch=" << maxBatch << endl;

    unsigned ind = 0;
    while (ind < maxBatch) {
      SentencePtr sentence = coll_[ind];
      unsigned sentLen = sentence->GetWords(0).size();

      if (sentences->size() && (numWords + sentLen) > batchWords) {
        // max batch
        break;
      }

      numWords += sentLen;

      // add next 32 sentences
      unsigned endInd = std::min(size(), ind + 32);
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
    unsigned startInd = (batchsize > size()) ? 0 : size() - batchsize;
    for (unsigned i = startInd; i < size(); ++i) {
      SentencePtr sentence = coll_[i];
      sentences->push_back(sentence);
    }

    coll_.resize(startInd);
  }

  return sentences;
}

std::string Sentences::Debug(unsigned verbosity) const
{
  std::stringstream strm;
  for (unsigned i = 0; i < size(); ++i) {
    SentencePtr sent = at(i);
    strm << sent->Debug(verbosity) << std:: endl;;
  }

  return strm.str();
}


}

