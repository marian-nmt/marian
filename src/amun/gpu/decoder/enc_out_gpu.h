#pragma once

#include "../mblas/matrix.h"
#include "common/enc_out.h"

namespace amunmt {
namespace GPU {

class EncOutGPU : public EncOut
{
public:
  EncOutGPU(SentencesPtr sentences);

  mblas::Matrix &GetSourceContext()
  { return sourceContext_; }

  const mblas::Matrix &GetSourceContext() const
  { return sourceContext_; }

protected:
  mblas::Matrix sourceContext_;

  mblas::Matrix states_;
  mblas::Matrix embeddings_;

  mblas::Matrix SCU_;

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
