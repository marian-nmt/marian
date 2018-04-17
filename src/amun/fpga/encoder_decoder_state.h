#pragma once
#include "common/scorer.h"
#include "matrix.h"

namespace amunmt {
namespace FPGA {

class EncoderDecoderState : public State {
public:
  EncoderDecoderState(const OpenCLInfo &openCLInfo);

  mblas::Tensor& GetStates();
  mblas::Tensor& GetEmbeddings();
  const mblas::Tensor& GetStates() const;
  const mblas::Tensor& GetEmbeddings() const;

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  mblas::Tensor states_;
  mblas::Tensor embeddings_;

};

}
}


