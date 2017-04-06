#pragma once

#include "model.h"
#include "matrix.h"
#include "gru.h"

namespace amunmt {

class God;

namespace FPGA {

template<typename T>
class Array;

class Decoder {

  template <class Weights>
  class Embeddings {
  public:
    Embeddings(const Weights& model)
    : w_(model)
    {}

  private:
    const Weights& w_;

  };

  template <class Weights1, class Weights2>
  class RNNHidden {
    public:
    RNNHidden(const OpenCLInfo &openCLInfo, const Weights1& initModel, const Weights2& gruModel)
    : w_(initModel), gru_(openCLInfo, gruModel) {}

    private:
      const Weights1& w_;
      const GRU<Weights2> gru_;

  };

  template <class Weights>
  class RNNFinal {
    public:
    RNNFinal(const OpenCLInfo &openCLInfo, const Weights& model)
    : gru_(openCLInfo, model) {}

    private:
      const GRU<Weights> gru_;
  };

  template <class Weights>
  class Alignment {
    public:
    Alignment(const God &god, const Weights& model)
      : w_(model)
    {}

    private:
      const Weights& w_;

  };

  template <class Weights>
  class Softmax {
  public:
    Softmax(const Weights& model)
    : w_(model), filtered_(false)
    {
    }

  private:
    const Weights& w_;
    bool filtered_;

  };

public:
  Decoder(const OpenCLInfo &openCLInfo, const God &god, const Weights& model)
  : embeddings_(model.decEmbeddings_),
    rnn1_(openCLInfo, model.decInit_, model.decGru1_),
    rnn2_(openCLInfo, model.decGru2_),
    alignment_(god, model.decAlignment_),
    softmax_(model.decSoftmax_)
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

private:
  Embeddings<Weights::DecEmbeddings> embeddings_;
  RNNHidden<Weights::DecInit, Weights::DecGRU1> rnn1_;
  RNNFinal<Weights::DecGRU2> rnn2_;
  Alignment<Weights::DecAlignment> alignment_;
  Softmax<Weights::DecSoftmax> softmax_;

};

}
}
