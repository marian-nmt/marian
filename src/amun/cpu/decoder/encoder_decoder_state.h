#pragma once

#include <vector>

#include "cpu/mblas/tensor.h"
#include "common/scorer.h"

namespace amunmt {
namespace CPU {

class EncoderDecoderState : public State {
  public:
    EncoderDecoderState();
    EncoderDecoderState(const EncoderDecoderState&) = delete;

    virtual std::string Debug(unsigned verbosity = 1) const;

    CPU::mblas::Tensor& GetStates();
    const CPU::mblas::Tensor& GetStates() const;

  	CPU::mblas::Tensor& GetEmbeddings();
    const CPU::mblas::Tensor& GetEmbeddings() const;

  private:
    CPU::mblas::Tensor states_;
    CPU::mblas::Tensor embeddings_;
};

}  // namespace CPU
}  // namespace amunmt
