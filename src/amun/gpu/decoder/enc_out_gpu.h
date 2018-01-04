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

  const mblas::Vector<unsigned> &GetSentenceLengths() const
  { return sentenceLengths_; }

  mblas::Matrix &GetSCU()
  { return SCU_; }

  const mblas::Matrix &GetSCU() const
  { return SCU_; }

protected:
  mblas::Matrix sourceContext_, SCU_;
  mblas::Vector<unsigned> sentenceLengths_;

};


/////////////////////////////////////////////////////////////////////////

}
}
