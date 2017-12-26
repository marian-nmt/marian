#pragma once
#include <memory>
#include <vector>
#include <stddef.h>
#include <cassert>
#include "god.h"
#include "history.h"

namespace amunmt {

class Sentences;
class History;
class BufferOutput;

class Hypothesis;
using HypothesisPtr = std::shared_ptr<Hypothesis>;
using Hypotheses = std::vector<HypothesisPtr>;
using HypothesesBatch = std::vector<Hypotheses>;


class HistoriesElement
{
public:
  HistoriesElement(const SentencePtr &sentence, bool normalizeScore);

  unsigned GetBeamSize() const
  { return beamSize_; }

  void SetNewBeamSize(unsigned val);

  const History &GetHistory() const
  { return history_; }

  const SentencePtr &GetSentence() const
  { return sentence_; }

  void Add(const Hypotheses &hypos, Hypotheses &survivors);

  bool IsFirst() const;

protected:
  unsigned beamSize_;  // beam size 0..beam
  History history_;
  SentencePtr sentence_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////
using HistoriesElementPtr = std::shared_ptr<HistoriesElement>;
///////////////////////////////////////////////////////////////////////////////////////////////////

class Histories
{
public:
  Histories(bool normalizeScore);

  void Init(const std::vector<BufferOutput> &newSentences);

  size_t size() const
  { return coll_.size(); }

  unsigned GetNumActive() const
  { return active_; }

  bool NormalizeScore() const
  { return normalizeScore_; }

  const HistoriesElementPtr &Get(size_t ind) const
  { return coll_[ind]; }

  HistoriesElementPtr &Get(size_t ind)
  { return coll_[ind]; }

  void Set(size_t ind, HistoriesElement *val);

  size_t GetBeamSize(size_t ind) const;
  bool Empty(size_t ind) const;

  size_t Sum() const;

  size_t MaxLength() const;

  void SetNewBeamSize(unsigned val);

  std::vector<unsigned> GetBeamSizes() const;
  std::vector<char> IsFirsts() const;

  Hypotheses Add(const God &god, const HypothesesBatch& beams);
  Hypotheses GetFirstHyps();
  void OutputAll(const God &god);

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  bool normalizeScore_;
  std::vector<HistoriesElementPtr> coll_;
  unsigned active_;
};

}
