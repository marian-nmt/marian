#include "enc_out.h"

using namespace std;

namespace amunmt {

EncOut::EncOut(const SentencesPtr &sentences)
:sentences_(sentences)
,h_sentenceLengths_(sentences->size())
{
  size_t tab = 0;

  for (size_t i = 0; i < sentences->size(); ++i) {
    h_sentenceLengths_[i] = sentences->Get(i)->GetWords(tab).size();
  }
}

EncOut::~EncOut()
{
  //cerr << "~EncOut" << endl;
}


/////////////////////////////////////////////////////////////////////////////

BufferOutput::BufferOutput(EncOutPtr vencOut, size_t vSentenceOffset)
:encOut_(vencOut)
,sentenceOffset_(vSentenceOffset)
{}

const SentencePtr &BufferOutput::GetSentence() const
{
  const Sentences &sentences = encOut_->GetSentences();
  const SentencePtr &sentence = sentences.Get(sentenceOffset_);
  return sentence;
}


}

