#include <numeric>
#include "beam_size.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(size_t s, size_t val)
:sentences_(s)
{
  for (size_t i = 0; i < size(); ++i) {
    sentences_[i].size = val;
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

}
