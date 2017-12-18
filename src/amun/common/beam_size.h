#pragma once
#include <memory>
#include <vector>
#include <stddef.h>
#include <cassert>
#include "beam.h"

namespace amunmt {

class Sentences;
class History;

class BeamElement
{
public:
  BeamElement()
  {}

  BeamElement(unsigned size, History *history);

  unsigned GetBeamSize() const
  { return size_; }

  void SetBeamSize(unsigned size)
  {
    size_ = size;
  }

  void Decr()
  {
    assert(size_);
    --size_;
  }

  const History &GetHistory() const
  { return *history_; }

  History &GetHistory()
  { return *history_; }

protected:
  unsigned size_;  // beam size 0..beam
  std::shared_ptr<History> history_;

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
  void Decr(size_t ind);

  size_t Sum() const;

  std::vector<size_t> Vec() const;

  void Add(const Beams& beams);
  Beam GetFirstHyps();
  void Output(const God &god) const;

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  std::vector<BeamElement> coll_;

};

}
