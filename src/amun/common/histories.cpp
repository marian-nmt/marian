#include <sstream>
#include <numeric>
#include "histories.h"
#include "sentences.h"
#include "history.h"
#include "enc_out.h"
#include "utils.h"

using namespace std;

namespace amunmt {

HistoriesElement::HistoriesElement(const SentencePtr &sentence, bool normalizeScore)
:beamSize_(1)
,history_(*sentence, normalizeScore, 3 * sentence->size())
,sentence_(sentence)
{
  const Hypotheses &hypos = history_.front();
  assert(hypos.size() == 1);
  hypos_.push_back(hypos[0]);
}

void HistoriesElement::Add()
{
  Hypotheses survivors = history_.Add(hypos_);
  unsigned numEOS = hypos_.size() - survivors.size();
  hypos_ = survivors;
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

void HistoriesElement::StartCalcBeam()
{
  hypos_.clear();
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

size_t Histories::GetTotalBeamSize() const
{
  size_t ret = 0;
  for (size_t i = 0; i < size(); ++i) {
    ret += GetBeamSize(i);
  }
  return ret;
}

size_t Histories::NumCandidates() const
{
  size_t ret = 0;
  for (size_t i = 0; i < size(); ++i) {
    const HistoriesElementPtr &ele = coll_[i];
    if (ele) {
      size_t beamSize = ele->GetBeamSize();
      if (ele->IsFirst()) {
        ret += beamSize;
      }
      else {
        ret += beamSize * beamSize;
      }
    }
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

void Histories::Add(const God &god)
{
  for (size_t i = 0; i < size(); ++i) {
    HistoriesElementPtr &ele = Get(i);

    if (ele) {
      /*
      cerr << "hypos="
          << hypos.size() << " "
          << coll_[i]->GetBeamSize()
          << endl;
      */
      ele->Add();
      unsigned beamSize = ele->GetBeamSize();

      if (beamSize == 0) {
        ele->GetHistory().Output(god);
        ele.reset();

        assert(active_);
        --active_;
      }
    }
  }

}

void Histories::StartCalcBeam()
{
  for (size_t i = 0; i < size(); ++i) {
    HistoriesElementPtr &ele = Get(i);

    if (ele) {
      ele->StartCalcBeam();
    }
  }
}

Hypotheses Histories::GetSurvivors() const
{
  Hypotheses ret;
  for (size_t i = 0; i < coll_.size(); ++i) {
    HistoriesElementPtr ele =  coll_[i];
    if (ele) {
      const Hypotheses &hypos = ele->GetHypotheses();
      //ret.insert(ret.end(), hypos.begin(), hypos.end());
      std::copy (hypos.begin(), hypos.end(), std::back_inserter(ret));
    }
  }
  return ret;
}

std::vector<unsigned> Histories::GetWords() const
{
  Hypotheses survivors = GetSurvivors();
  std::vector<unsigned> ret(survivors.size());

  for (size_t i = 0; i < survivors.size(); ++i) {
    const HypothesisPtr &hypo = survivors[i];
    unsigned word = hypo->GetWord();
    ret[i] = word;
  }

  return ret;
}

std::vector<unsigned> Histories::GetPrevStateIndices() const
{
  Hypotheses survivors = GetSurvivors();
  std::vector<unsigned> ret(survivors.size());

  for (size_t i = 0; i < survivors.size(); ++i) {
    const HypothesisPtr &hypo = survivors[i];
    unsigned word = hypo->GetPrevStateIndex();
    ret[i] = word;
  }

  return ret;
}

std::vector<unsigned> Histories::Hypo2Batch() const
{
  std::vector<unsigned> ret;
  for (size_t i = 0; i < coll_.size(); ++i) {
    const HistoriesElementPtr &ele = coll_[i];
    if (ele) {
      for (size_t j = 0; j < ele->GetBeamSize(); ++j) {
        ret.push_back(i);
      }
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

  strm << " size=" << coll_.size() << " ";

  if (verbosity) {
    strm << "beamsize=";
    for (size_t i = 0; i < coll_.size(); ++i) {
      const HistoriesElementPtr &ele = coll_[i];
      if (ele) {
        //strm << " (" << ele.sentenceInd << "," << ele.size << ")";
        strm << "(" << i
              << ",sent=" << ele->GetSentence()->GetLineNum()
              << ",len=" << ele->GetSentence()->size()
              << ",beam=" << ele->GetBeamSize()
              << ") ";
      }
      else {
        strm << "NULL ";
      }
    }

    strm << "newBatchIds_=" << amunmt::Debug(newBatchIds_, 2);
  }

  return strm.str();
}

void Histories::StartTopup()
{
  newSentenceLengths_.clear();
  newBatchIds_.clear();
  nextBatchInd_ = 0;
}

void Histories::Topup(HistoriesElement *val)
{
  unsigned ind = FindNextEmptyIndex();
  Set(ind, val);
  newBatchIds_.push_back(ind);
}

const std::vector<unsigned> &Histories::GetNewSentenceLengths() const
{
  if (newSentenceLengths_.size() == 0) {
    newSentenceLengths_.resize(newBatchIds_.size());
    for (size_t i = 0; i < newBatchIds_.size(); ++i) {
      unsigned ind = newBatchIds_[i];
      const HistoriesElementPtr &ele = Get(ind);
      assert(ele);
      const SentencePtr &sent = ele->GetSentence();
      assert(sent);

      newSentenceLengths_[i] = sent->size();
    }
  }
  return newSentenceLengths_;
}

unsigned Histories::FindNextEmptyIndex()
{
  while(nextBatchInd_ < size()) {
    const HistoriesElementPtr &ele = Get(nextBatchInd_);
    if (ele == nullptr) {
      return nextBatchInd_++;
    }
    else {
      ++nextBatchInd_;
    }
  }

  assert(false);
  return 9999999;;
}

}
