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


class Beam
{
public:
  Beam(unsigned size, const Sentence &sentence, bool normalizeScore, size_t maxLength);

  unsigned GetBeamSize() const
  { return size_; }

  void SetBeamSize(unsigned size)
  {
    size_ = size;
  }

  const History &GetHistory() const
  { return history_; }

  void Add(const Hypotheses &hypos, Hypotheses &survivors);

protected:
  unsigned size_;  // beam size 0..beam
  History history_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////

class Beams
{
public:
  Beams(const Sentences& sentences, size_t val, bool normalizeScore);

  size_t size() const
  { return coll_.size(); }

  unsigned GetNumActive() const
  { return active_; }

  size_t GetBeamSize(size_t ind) const;
  void SetBeamSize(size_t ind, size_t val);
  bool Empty(size_t ind) const;

  size_t Sum() const;

  std::vector<size_t> GetBeamSizes() const;

  Hypotheses Add(const God &god, const HypothesesBatch& beams);
  Hypotheses GetFirstHyps();
  void OutputAll(const God &god);

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector< std::shared_ptr<Beam> > coll_;
  unsigned active_;
};

}
