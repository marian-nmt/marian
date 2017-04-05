#pragma once

#include "model.h"
#include "matrix.h"

namespace amunmt {
namespace FPGA {

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

};

}
}
