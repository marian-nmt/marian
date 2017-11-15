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

    virtual std::string Debug(size_t verbosity = 1) const;

    CellState& GetStates();
    mblas::Matrix& GetEmbeddings();
    const CellState& GetStates() const;
    const mblas::Matrix& GetEmbeddings() const;

  private:
    CellState states_;
    mblas::Matrix embeddings_;
};

}
}  // namespace GPU
