#include <sstream>
#include <numeric>
#include "histories.h"
#include "sentences.h"
#include "history.h"
#include "enc_out.h"

using namespace std;

namespace amunmt {

HistoriesElement::HistoriesElement(const SentencePtr &sentence, bool normalizeScore)
:beamSize_(1)
,history_(*sentence, normalizeScore, 3 * sentence->size())
,sentence_(sentence)
{}

void HistoriesElement::Add(const Hypotheses &hypos, Hypotheses &survivors)
{
  unsigned numEOS = history_.Add(hypos, survivors);
  assert(beamSize_ >= numEOS);
  beamSize_ -= numEOS;
}

void HistoriesElement::SetNewBeamSize(unsigned val)
{
  if (IsFirst()) {
    beamSize_ = val;
  }
}

bool HistoriesElement::IsFirst() const
{
  return history_.size() == 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
Histories::Histories(bool normalizeScore)
:normalizeScore_(normalizeScore)
,active_(0)
{
}

void Histories::Init(const std::vector<BufferOutput> &newSentences)
{
  coll_.resize(newSentences.size());

  for (size_t i = 0; i < size(); ++i) {
    const SentencePtr &sentence = newSentences[i].GetSentence();
    if (sentence) {
      Set(i, new HistoriesElement(sentence, normalizeScore_));
    }
  }
}

void Histories::Set(size_t ind, HistoriesElement *val)
{
  HistoriesElementPtr &ele = coll_[ind];
  assert(ele == nullptr);
  ele.reset(val);
  ++active_;
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
  for (size_t i = 0; i < coll_.size(); ++i) {
    ret += GetBeamSize(i);
  }

  return ret;
}

size_t Histories::MaxLength() const
{
  size_t ret = 0;
  for (size_t i = 0; i < coll_.size(); ++i) {
    const HistoriesElementPtr &ele = coll_[i];
    if (ele) {
      const SentencePtr &sent = ele->GetSentence();
      size_t size = sent->size();
      if (ret < size) {
        ret = size;
      }
    }
  }

  return ret;
}

std::vector<unsigned> Histories::GetBeamSizes() const
{
  std::vector<unsigned> ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret[i] = GetBeamSize(i);
  }
  return ret;
}

std::vector<char> Histories::IsFirsts() const
{
  std::vector<char> ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret[i] = Empty(i) ? false : coll_[i]->IsFirst();
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
    HistoriesElementPtr ele =  coll_[i];
    if (ele) {
      const History &history = ele->GetHistory();
      const Hypotheses &beam = history.front();
      HypothesisPtr hypo = beam[0];
      ret[i] = hypo;
    }
  }
  return ret;
}

void Histories::OutputAll(const God &god)
{
  for (size_t i = 0; i < coll_.size(); ++i) {
    HistoriesElementPtr &ele = coll_[i];
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
