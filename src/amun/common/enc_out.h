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

  template<class T>
  T &GetStates()
  { return static_cast<T&>(GetStatesInternal()); }

  template<class T>
  const T &GetStates() const
  { return static_cast<const T&>(GetStatesInternal()); }

  template<class T>
  T &GetEmbeddings()
  { return static_cast<T&>(GetEmbeddingsInternal()); }

  template<class T>
  const T &GetEmbeddings() const
  { return static_cast<const T&>(GetEmbeddingsInternal()); }

  template<class T>
  T &GetSCU()
  { return static_cast<T&>(GetSCUInternal()); }

  template<class T>
  const T &GetSCU() const
  { return static_cast<const T&>(GetSCUInternal()); }

protected:
  SentencesPtr sentences_;
  std::vector<uint> h_sentenceLengths_;

  virtual BaseMatrix &GetStatesInternal() = 0;
  virtual const BaseMatrix &GetStatesInternal() const = 0;

  virtual BaseMatrix &GetEmbeddingsInternal() = 0;
  virtual const BaseMatrix &GetEmbeddingsInternal() const = 0;

  virtual BaseMatrix &GetSCUInternal() = 0;
  virtual const BaseMatrix &GetSCUInternal() const = 0;
};

}
