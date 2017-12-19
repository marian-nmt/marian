#include <sstream>
#include <numeric>
#include "histories.h"
#include "sentences.h"
#include "history.h"

using namespace std;

namespace amunmt {

HistoriesElement::HistoriesElement(unsigned size, const Sentence &sentence, bool normalizeScore, size_t maxLength)
:size_(size)
,history_(sentence, normalizeScore, 3 * sentence.size())
{}

void HistoriesElement::Add(const Hypotheses &hypos, Hypotheses &survivors)
{
  unsigned numEOS = history_.Add(hypos, survivors);
  assert(size_ >= numEOS);
  size_ -= numEOS;

}

void HistoriesElement::SetNewBeamSize(unsigned val)
{
  if (history_.size() == 1) {
    size_ = val;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

Histories::Histories(const Sentences& sentences, size_t val, bool normalizeScore)
:coll_(sentences.size())
,active_(sentences.size())
{
  for (size_t i = 0; i < size(); ++i) {
    const Sentence &sentence = sentences.Get(i);
    coll_[i].reset(new HistoriesElement(val, sentence, normalizeScore, 3 * sentence.size()));
  }
}

size_t Histories::GetBeamSize(size_t ind) const
{
  return Empty(ind) ? 0 : coll_[ind]->GetBeamSize();
}

bool Histories::Empty(size_t ind) const
{
  return coll_[ind] == nullptr;
}

size_t Histories::Sum() const
{
  size_t ret = 0;
  for (size_t i = 0; i < size(); ++i) {
    ret += GetBeamSize(i);
  }

  return ret;
}

std::vector<size_t> Histories::GetBeamSizes() const
{
  std::vector<size_t> ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret[i] = GetBeamSize(i);
  }
  return ret;
}

void Histories::SetNewBeamSize(unsigned val)
{
  for (size_t i = 0; i < size(); ++i) {
    if (!Empty(i)) {
      coll_[i]->SetNewBeamSize(val);
    }
  }

}

Hypotheses Histories::Add(const God &god, const HypothesesBatch& beams)
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
      std::shared_ptr<HistoriesElement> &ele = coll_[i];
      assert(ele);
      ele->Add(hypos, survivors);
      unsigned beamSize = ele->GetBeamSize();

      if (beamSize == 0) {
        ele->GetHistory().Output(god);
        ele.reset();

        assert(active_);
        --active_;
      }
    }
  }

  return survivors;
}

Hypotheses Histories::GetFirstHyps()
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

void Histories::OutputAll(const God &god)
{
  for (size_t i = 0; i < coll_.size(); ++i) {
    std::shared_ptr<HistoriesElement> &ele = coll_[i];
    if (ele && ele->GetBeamSize()) {
      const History &history = ele->GetHistory();
      history.Output(god);
      ele.reset();
    }
  }

  active_ = 0;
}

std::string Histories::Debug(size_t verbosity) const
{
  stringstream strm;

  strm << " coll_=" << coll_.size();
  //strm << "total_=" << total_;
  //strm << " maxLength_=" << maxLength_;

  /*
  if (verbosity) {
    uint sum = 0;
    for (size_t i = 0; i < coll_.size(); ++i) {
      const HistoriesElement &ele = coll_[i];
      strm << " (" << ele.sentenceInd << "," << ele.size << ")";

      sum += ele.size;
    }
    assert(sum == total_);
  }
  */
  return strm.str();
}

}
