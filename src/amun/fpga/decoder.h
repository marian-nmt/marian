#pragma once

#include "model.h"
#include "matrix.h"

namespace amunmt {

class God;

namespace FPGA {

template<typename T>
class Array;

class Decoder {

public:
  Decoder(const God &god, const Weights& model)
  {}

  size_t GetVocabSize() const {
  }

  mblas::Matrix& GetProbs() {
  }

  mblas::Matrix& GetAttention() {
  }

  void EmptyState(mblas::Matrix& State,
                  const mblas::Matrix& SourceContext,
                  size_t batchSize,
                  const Array<int>& batchMapping);

};

}
}
