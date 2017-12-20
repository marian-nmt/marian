#pragma once

#include "../mblas/matrix.h"
#include "common/enc_out.h"

namespace amunmt {
namespace GPU {

class EncOutGPU : public EncOut
{
public:
  EncOutGPU(SentencesPtr sentences);
  ~EncOutGPU();

  mblas::Matrix &GetSourceContext()
  { return sourceContext_; }

  const mblas::Matrix &GetSourceContext() const
  { return sourceContext_; }

  const mblas::Vector<uint> &GetSentenceLengths() const
  { return sentenceLengths_; }

protected:
  mblas::Matrix sourceContext_;
  mblas::Vector<uint> sentenceLengths_;

};


/////////////////////////////////////////////////////////////////////////

}
}
