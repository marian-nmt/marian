#pragma once

#include <vector>

#include "cpu/mblas/matrix.h"
#include "common/scorer.h"

namespace amunmt {
namespace CPU {

class EncoderDecoderState : public State {
  public:
    EncoderDecoderState();
    EncoderDecoderState(const EncoderDecoderState&) = delete;

    virtual std::string Debug(size_t verbosity = 1) const;

    CPU::mblas::Matrix& GetStates();
    const CPU::mblas::Matrix& GetStates() const;

  	CPU::mblas::Matrix& GetEmbeddings();
    const CPU::mblas::Matrix& GetEmbeddings() const;

  private:
    CPU::mblas::Matrix states_;
    CPU::mblas::Matrix embeddings_;
};

}  // namespace CPU
}  // namespace amunmt
