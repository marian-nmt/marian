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


class BeamElement
{
public:
  BeamElement(unsigned size, const Sentence &sentence, bool normalizeScore, size_t maxLength);

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

class BeamSize
{
public:
  BeamSize(const Sentences& sentences, size_t val, bool normalizeScore);

  size_t size() const
  { return coll_.size(); }

  size_t Get(size_t ind) const;
  void Set(size_t ind, size_t val);

  size_t Sum() const;

  std::vector<size_t> Vec() const;

  Hypotheses Add(const HypothesesBatch& beams);
  Hypotheses GetFirstHyps();
  void Output(const God &god) const;

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector< std::shared_ptr<BeamElement> > coll_;

};

}
