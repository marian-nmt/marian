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

class Hypothesis;
using HypothesisPtr = std::shared_ptr<Hypothesis>;
using Hypotheses = std::vector<HypothesisPtr>;
using HypothesesBatch = std::vector<Hypotheses>;


class HistoriesElement
{
public:
  HistoriesElement(unsigned size, const Sentence &sentence, bool normalizeScore, size_t maxLength);

  unsigned GetBeamSize() const
  { return size_; }

  void SetNewBeamSize(unsigned val);

  const History &GetHistory() const
  { return history_; }

  void Add(const Hypotheses &hypos, Hypotheses &survivors);

  bool IsFirst() const;

protected:
  unsigned size_;  // beam size 0..beam
  History history_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////
using HistoriesElementPtr = std::shared_ptr<HistoriesElement>;
///////////////////////////////////////////////////////////////////////////////////////////////////

class Histories
{
public:
  Histories(const Sentences& sentences, size_t val, bool normalizeScore);

  size_t size() const
  { return coll_.size(); }

  unsigned GetNumActive() const
  { return active_; }

  const HistoriesElementPtr &Get(size_t ind) const
  { return coll_[ind]; }

  size_t GetBeamSize(size_t ind) const;
  bool Empty(size_t ind) const;

  size_t Sum() const;

  void SetNewBeamSize(unsigned val);

  std::vector<unsigned> GetBeamSizes() const;
  std::vector<unsigned> GetIsFirsts() const;

  Hypotheses Add(const God &god, const HypothesesBatch& beams);
  Hypotheses GetFirstHyps();
  void OutputAll(const God &god);

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<HistoriesElementPtr> coll_;
  unsigned active_;
};

}
