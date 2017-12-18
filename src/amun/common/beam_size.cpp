#include <numeric>
#include "beam_size.h"
#include "sentences.h"
#include "histories.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(const Sentences& sentences, size_t val, bool normalizeScore)
:sentences_(sentences.size())
{
  for (size_t i = 0; i < size(); ++i) {
    sentences_[i].size = val;

    const Sentence &sentence = sentences.Get(i);
    History *history = new History(sentence, normalizeScore, 3 * sentence.size());
    sentences_[i].history.reset(history);
  }
}

size_t BeamSize::Get(size_t ind) const
{
  return sentences_[ind].size;
}

void BeamSize::Set(size_t ind, size_t val)
{
  sentences_[ind].size = val;
}

void BeamSize::Decr(size_t ind)
{
  sentences_[ind].Decr();
}

size_t BeamSize::Sum() const
{
  size_t ret = 0;
  for (size_t i = 0; i < size(); ++i) {
    ret += sentences_[i].size;
  }

  return ret;
}

std::vector<size_t> BeamSize::Vec() const
{
  std::vector<size_t> ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret[i] = sentences_[i].size;
  }
  return ret;
}

void BeamSize::Add(const Beams& beams) {
  for (size_t i = 0; i < size(); ++i) {
    if (!beams[i].empty()) {
      sentences_[i].history->Add(beams[i]);
    }
  }
}

Beam BeamSize::GetFirstHyps()
{
  Beam ret(sentences_.size());
  for (size_t i = 0; i < sentences_.size(); ++i) {
    History &history = *sentences_[i].history;
    Beam &beam = history.front();
    HypothesisPtr hypo = beam[0];
    ret[i] = hypo;
  }
  return ret;
}

void BeamSize::Output(const God &god) const
{
  for (size_t i = 0; i < sentences_.size(); ++i) {
    const History &history = *sentences_.at(i).history;
    history.Output(god);

  }
}

}
