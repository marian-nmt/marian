#pragma once

#include <string>
#include "common/scorer.h"

#include "gpu/mblas/matrix.h"

namespace GPU {
class EncoderDecoderState : public State {
  public:
    virtual std::string Debug() const;

    mblas::Matrix& GetStates();
    mblas::Matrix& GetEmbeddings();
    const mblas::Matrix& GetStates() const;
    const mblas::Matrix& GetEmbeddings() const;

  private:
    mblas::Matrix states_;
    mblas::Matrix embeddings_;
};
}
