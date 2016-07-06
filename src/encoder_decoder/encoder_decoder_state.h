#pragma once

#include "matrix.h"
#include "decoder/scorer.h"

class EncoderDecoderState : public State {
  public:
    mblas::Matrix& GetStates() {
      return states_;
    }

    mblas::Matrix& GetEmbeddings() {
      return embeddings_;
    }

    const mblas::Matrix& GetStates() const {
      return states_;
    }

    const mblas::Matrix& GetEmbeddings() const {
      return embeddings_;
    }

  private:
    mblas::Matrix states_;
    mblas::Matrix embeddings_;
};
