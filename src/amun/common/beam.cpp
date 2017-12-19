#include <sstream>
#include <numeric>
#include "beam.h"
#include "sentences.h"
#include "history.h"

using namespace std;

namespace amunmt {

Beam::Beam(unsigned size, const Sentence &sentence, bool normalizeScore, size_t maxLength)
:size_(size)
,history_(sentence, normalizeScore, 3 * sentence.size())
{}

void Beam::Add(const Hypotheses &hypos, Hypotheses &survivors)
{
  unsigned numEOS = history_.Add(hypos, survivors);
  assert(size_ >= numEOS);
  size_ -= numEOS;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

Beams::Beams(const Sentences& sentences, size_t val, bool normalizeScore)
:coll_(sentences.size())
{
  for (size_t i = 0; i < size(); ++i) {
    const Sentence &sentence = sentences.Get(i);
    coll_[i].reset(new Beam(val, sentence, normalizeScore, 3 * sentence.size()));
  }
}

size_t Beams::GetBeamSize(size_t ind) const
{
  return Empty(ind) ? 0 : coll_[ind]->GetBeamSize();
}

void Beams::SetBeamSize(size_t ind, size_t val)
{
  if (!Empty(ind)) {
    coll_[ind]->SetBeamSize(val);
  }
}

bool Beams::Empty(size_t ind) const
{
  return coll_[ind] == nullptr;
}

size_t Beams::Sum() const
{
  size_t ret = 0;
  for (size_t i = 0; i < size(); ++i) {
    ret += GetBeamSize(i);
  }

  return ret;
}

std::vector<size_t> Beams::Vec() const
{
  std::vector<size_t> ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret[i] = GetBeamSize(i);
  }
  return ret;
}

Hypotheses Beams::Add(const God &god, const HypothesesBatch& beams)
{
  Hypotheses survivors;

  for (size_t i = 0; i < size(); ++i) {
    const Hypotheses &hypos = beams[i];
    /*
    cerr << "hypos="
        << hypos.size() << " "
        << coll_[i]->GetBeamSize()
        << endl;
    */
    if (hypos.size()) {
      std::shared_ptr<Beam> &ele = coll_[i];
      assert(ele);
      ele->Add(hypos, survivors);
      unsigned beamSize = ele->GetBeamSize();

      if (beamSize == 0) {
        ele->GetHistory().Output(god);
        ele.reset();
      }
    }
  }

  return survivors;
}

Hypotheses Beams::GetFirstHyps()
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

void Beams::OutputAll(const God &god) const
{
  for (size_t i = 0; i < coll_.size(); ++i) {
    const std::shared_ptr<Beam> &ele = coll_[i];
    if (ele && ele->GetBeamSize()) {
      const History &history = ele->GetHistory();
      history.Output(god);
    }
  }
}

std::string Beams::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << " coll_=" << coll_.size();
  //strm << "total_=" << total_;
  //strm << " maxLength_=" << maxLength_;

  /*
  if (verbosity) {
    uint sum = 0;
    for (size_t i = 0; i < coll_.size(); ++i) {
      const Beam &ele = coll_[i];
      strm << " (" << ele.sentenceInd << "," << ele.size << ")";

      sum += ele.size;
    }
    assert(sum == total_);
  }
  */
  return strm.str();
}

}
