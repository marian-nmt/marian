#pragma once

#include "common/enc_out.h"
#include "../mblas/matrix.h"
#include "../dl4mt/cellstate.h"

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

  mblas::Vector<unsigned> &GetSentenceLengths()
  { return sentenceLengths_; }

  const mblas::Vector<unsigned> &GetSentenceLengths() const
  { return sentenceLengths_; }

  mblas::Matrix &GetSCU()
  { return SCU_; }

  const mblas::Matrix &GetSCU() const
  { return SCU_; }

  CellState &GetCellState()
  { return state_; }

  const CellState &GetCellState() const
  { return state_; }

protected:
  mblas::Matrix sourceContext_, SCU_;
  mblas::Vector<unsigned> sentenceLengths_;
  CellState state_;

};


/////////////////////////////////////////////////////////////////////////

}
}
