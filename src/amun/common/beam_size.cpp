#include <sstream>
#include <numeric>
#include "beam_size.h"
#include "sentences.h"
#include "history.h"

using namespace std;

namespace amunmt {

BeamElement::BeamElement(unsigned size, History *history)
:size_(size)
,history_(history)
{}

void BeamElement::Add(const Hypotheses &hypos, Hypotheses &survivors)
{

  history_->Add(hypos);

  for (const HypothesisPtr &h : hypos) {
    if (h->GetWord() != EOS_ID) {
      survivors.push_back(h);
    }
    else {
      Decr();
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

BeamSize::BeamSize(const Sentences& sentences, size_t val, bool normalizeScore)
:coll_(sentences.size())
{
  for (size_t i = 0; i < size(); ++i) {
    const Sentence &sentence = sentences.Get(i);
    History *history = new History(sentence, normalizeScore, 3 * sentence.size());

    coll_[i] = BeamElement(val, history);
  }
}

size_t BeamSize::Get(size_t ind) const
{
  return coll_[ind].GetBeamSize();
}

void BeamSize::Set(size_t ind, size_t val)
{
  coll_[ind].SetBeamSize(val);
}

void BeamSize::Decr(size_t ind)
{
  coll_[ind].Decr();
}

size_t BeamSize::Sum() const
{
  size_t ret = 0;
  for (size_t i = 0; i < size(); ++i) {
    ret += coll_[i].GetBeamSize();
  }

  return ret;
}

std::vector<size_t> BeamSize::Vec() const
{
  std::vector<size_t> ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret[i] = coll_[i].GetBeamSize();
  }
  return ret;
}

Hypotheses BeamSize::Add(const HypothesesBatch& beams)
{
  Hypotheses survivors;

  for (size_t i = 0; i < size(); ++i) {
    const Hypotheses &hypos = beams[i];
    coll_[i].Add(hypos, survivors);
  }

  return survivors;
}

Hypotheses BeamSize::GetFirstHyps()
{
  Hypotheses ret(coll_.size());
  for (size_t i = 0; i < coll_.size(); ++i) {
    History &history = coll_[i].GetHistory();
    Hypotheses &beam = history.front();
    HypothesisPtr hypo = beam[0];
    ret[i] = hypo;
  }
  return ret;
}

void BeamSize::Output(const God &god) const
{
  for (size_t i = 0; i < coll_.size(); ++i) {
    const History &history = coll_.at(i).GetHistory();
    history.Output(god);

  }
}

std::string BeamSize::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << " coll_=" << coll_.size();
  //strm << "total_=" << total_;
  //strm << " maxLength_=" << maxLength_;

  /*
  if (verbosity) {
    uint sum = 0;
    for (size_t i = 0; i < coll_.size(); ++i) {
      const BeamElement &ele = coll_[i];
      strm << " (" << ele.sentenceInd << "," << ele.size << ")";

      sum += ele.size;
    }
    assert(sum == total_);
  }
  */
  return strm.str();
}

}
