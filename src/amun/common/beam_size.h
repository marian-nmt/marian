#pragma once
#include <vector>
#include <stddef.h>

namespace amunmt {

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

  const std::vector<size_t> &Vec() const
  { return sentences_; }

protected:
  std::vector<size_t> sentences_;

};

}
