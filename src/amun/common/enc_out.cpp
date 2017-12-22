#include "enc_out.h"

using namespace std;

namespace amunmt {

SentenceElement::SentenceElement(EncOutPtr vencOut, size_t vsentenceInd)
:encOut(vencOut)
,sentenceInd(vsentenceInd)
{}

const SentencePtr &SentenceElement::GetSentence() const
{
  const Sentences &sentences = encOut->GetSentences();
  const SentencePtr &sentence = sentences.Get(sentenceInd);
  return sentence;
}

/////////////////////////////////////////////////////////////////////////////

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

}
