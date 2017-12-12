#pragma once
#include "base_matrix.h"
#include "sentence.h"
#include "sentences.h"

namespace amunmt {

class EncOut;
using EncOutPtr = std::shared_ptr<EncOut>;

class EncOut
{
public:

  /////////////////////////////////////////////////////////////////////////////
  struct SentenceElement
  {
    EncOutPtr encOut;
    size_t sentenceInd; // index of the sentence we're translation within encOut.sentences

    SentenceElement(EncOutPtr vencOut,
                    size_t vsentenceInd)
    :encOut(vencOut)
    ,sentenceInd(vsentenceInd)
    {}

    const Sentence &GetSentence() const
    {
      const Sentences &sentences = encOut->GetSentences();
      const Sentence &sentence = sentences.Get(sentenceInd);
      return sentence;
    }

  };
  /////////////////////////////////////////////////////////////////////////////

  EncOut(SentencesPtr sentences);

  template<class T>
  T &Get()
  { return static_cast<T&>(*this); }

  const Sentences &GetSentences() const
  { return *sentences_; }

  std::vector<uint> &GetSentenceLengthsHost()
  { return h_sentenceLengths_; }

  const std::vector<uint> &GetSentenceLengthsHost() const
  { return h_sentenceLengths_; }


protected:
  SentencesPtr sentences_;
  std::vector<uint> h_sentenceLengths_;

};

}
