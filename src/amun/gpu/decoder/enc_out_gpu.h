#pragma once

#include "../mblas/matrix.h"
#include "common/enc_out.h"

namespace amunmt {
namespace GPU {

class EncOutGPU : public EncOut
{
public:
  EncOutGPU(SentencesPtr sentences);


protected:
  mblas::Matrix sourceContext_;
  mblas::IMatrix sentenceLengths_;

  mblas::Matrix states_;
  mblas::Matrix embeddings_;

  mblas::Matrix SCU_;

  BaseMatrix &GetSourceContextInternal()
  { return sourceContext_; }

  const BaseMatrix &GetSourceContextInternal() const
  { return sourceContext_; }

  virtual const BaseMatrix &GetSentenceLengthsInternal() const
  { return sentenceLengths_; }

  BaseMatrix &GetStatesInternal()
  { return states_; }

  const BaseMatrix &GetStatesInternal() const
  { return states_; }

  BaseMatrix &GetEmbeddingsInternal()
  { return embeddings_; }

  const BaseMatrix &GetEmbeddingsInternal() const
  { return embeddings_; }

  BaseMatrix &GetSCUInternal()
  { return SCU_; }

  const BaseMatrix &GetSCUInternal() const
  { return SCU_; }

};


/////////////////////////////////////////////////////////////////////////

}
}
