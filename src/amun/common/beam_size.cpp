#include <numeric>
#include "beam_size.h"

using namespace std;

namespace amunmt {

BeamSize::BeamSize(size_t s, size_t val)
:sentences_(s, val)
{
}

size_t BeamSize::Get(size_t ind) const
{
  return sentences_[ind];
}

void BeamSize::Set(size_t ind, size_t val)
{
  sentences_[ind] = val;
}

void BeamSize::Decr(size_t ind)
{
  --sentences_[ind];
}

size_t BeamSize::Sum() const
{
  size_t beamSizeSum = std::accumulate(sentences_.begin(), sentences_.end(), 0);
  return beamSizeSum;
}

}
