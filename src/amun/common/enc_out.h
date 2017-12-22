#pragma once
#include "base_matrix.h"
#include "sentence.h"
#include "sentences.h"

namespace amunmt {

class EncOut
{
public:

  EncOut(const SentencesPtr &sentences);
  virtual ~EncOut();

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

/////////////////////////////////////////////////////////////////////////////
using EncOutPtr = std::shared_ptr<EncOut>;

/////////////////////////////////////////////////////////////////////////////

class BufferOutput
{
public:
  BufferOutput(EncOutPtr vencOut, size_t vsentenceInd);

  const SentencePtr &GetSentence() const;
  const EncOutPtr &GetEncOut() const
  { return encOut_; }

protected:
  EncOutPtr encOut_;
  size_t sentenceInd_; // index of the sentence we're translation within encOut.sentences

};

}
