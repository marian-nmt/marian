#pragma once
#include "common/scorer.h"
#include "matrix.h"

namespace amunmt {
namespace FPGA {

class EncoderDecoderState : public State {
public:
  EncoderDecoderState(const OpenCLInfo &openCLInfo);

  mblas::Matrix& GetStates();
  mblas::Matrix& GetEmbeddings();
  const mblas::Matrix& GetStates() const;
  const mblas::Matrix& GetEmbeddings() const;

  virtual std::string Debug(size_t verbosity = 1) const;

protected:
  mblas::Matrix states_;
  mblas::Matrix embeddings_;

};

}
}


