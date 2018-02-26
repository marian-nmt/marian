#pragma once

#include <string>
#include "common/scorer.h"

#include "gpu/dl4mt/cellstate.h"

namespace amunmt {
namespace GPU {

class EncoderDecoderState : public State {
  public:
	EncoderDecoderState(const EncoderDecoderState&) = delete;
	EncoderDecoderState() {}

    virtual std::string Debug(unsigned verbosity = 1) const;

    CellState& GetStates();
    mblas::Tensor& GetEmbeddings();
    const CellState& GetStates() const;
    const mblas::Tensor& GetEmbeddings() const;

  private:
    CellState states_;
    mblas::Tensor embeddings_;
};

}
}  // namespace GPU
