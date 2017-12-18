#pragma once
#include <vector>
#include <stddef.h>
#include <cassert>

namespace amunmt {

struct SentenceElement
{
  unsigned size;  // beam size 0..beam

  void Decr()
  {
    assert(size);
    --size;
  }

};

class BeamSize
{
public:
  BeamSize(size_t s, size_t val);

  size_t size() const
  { return sentences_.size(); }

  size_t Get(size_t ind) const;
  void Set(size_t ind, size_t val);
  void Decr(size_t ind);

  size_t Sum() const;

  std::vector<size_t> Vec() const;

protected:
  std::vector<SentenceElement> sentences_;

};

}
