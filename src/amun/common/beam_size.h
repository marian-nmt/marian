#pragma once
#include <memory>
#include <vector>
#include <stddef.h>
#include <cassert>
#include "beam.h"

namespace amunmt {

class Sentences;
class History;

struct SentenceElement
{
  unsigned size;  // beam size 0..beam
  std::shared_ptr<History> history;

  void Decr()
  {
    assert(size);
    --size;
  }

};

class BeamSize
{
public:
  BeamSize(const Sentences& sentences, size_t val, bool normalizeScore);

  size_t size() const
  { return sentences_.size(); }

  size_t Get(size_t ind) const;
  void Set(size_t ind, size_t val);
  void Decr(size_t ind);

  size_t Sum() const;

  std::vector<size_t> Vec() const;

  void Add(const Beams& beams);
  Beam GetFirstHyps();
  void Output(const God &god) const;

protected:
  std::vector<SentenceElement> sentences_;

};

}
