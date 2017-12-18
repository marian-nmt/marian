#include <sstream>
#include <numeric>
#include "beam_size.h"
#include "sentences.h"
#include "history.h"

using namespace std;

namespace amunmt {

BeamElement::BeamElement(unsigned size, const Sentence &sentence, bool normalizeScore, size_t maxLength)
:size_(size)
,history_(sentence, normalizeScore, 3 * sentence.size())
{}

void BeamElement::Add(const Hypotheses &hypos, Hypotheses &survivors)
{
  unsigned numEOS = history_.Add(hypos, survivors);
  assert(size_ >= numEOS);
  size_ -= numEOS;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

BeamSize::BeamSize(const Sentences& sentences, size_t val, bool normalizeScore)
:coll_(sentences.size())
{
  for (size_t i = 0; i < size(); ++i) {
    const Sentence &sentence = sentences.Get(i);
    coll_[i].reset(new BeamElement(val, sentence, normalizeScore, 3 * sentence.size()));
  }
}

size_t BeamSize::Get(size_t ind) const
{
  return coll_[ind]->GetBeamSize();
}

void BeamSize::Set(size_t ind, size_t val)
{
  coll_[ind]->SetBeamSize(val);
}

size_t BeamSize::Sum() const
{
  size_t ret = 0;
  for (size_t i = 0; i < size(); ++i) {
    ret += coll_[i]->GetBeamSize();
  }

  return ret;
}

std::vector<size_t> BeamSize::Vec() const
{
  std::vector<size_t> ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret[i] = coll_[i]->GetBeamSize();
  }
  return ret;
}

Hypotheses BeamSize::Add(const God &god, const HypothesesBatch& beams)
{
  Hypotheses survivors;

  for (size_t i = 0; i < size(); ++i) {
    const Hypotheses &hypos = beams[i];

    if (hypos.size()) {
      std::shared_ptr<BeamElement> &ele = coll_[i];
      ele->Add(hypos, survivors);
      unsigned beamSize = ele->GetBeamSize();

      if (beamSize == 0) {
        ele->GetHistory().Output(god);
      }
    }
  }

  return survivors;
}

Hypotheses BeamSize::GetFirstHyps()
{
  Hypotheses ret(coll_.size());
  for (size_t i = 0; i < coll_.size(); ++i) {
    const History &history = coll_[i]->GetHistory();
    const Hypotheses &beam = history.front();
    HypothesisPtr hypo = beam[0];
    ret[i] = hypo;
  }
  return ret;
}

void BeamSize::Output(const God &god) const
{
  for (size_t i = 0; i < coll_.size(); ++i) {
    const std::shared_ptr<BeamElement> &ele = coll_[i];
    if (ele->GetBeamSize()) {
      const History &history = ele->GetHistory();
      history.Output(god);
    }
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
