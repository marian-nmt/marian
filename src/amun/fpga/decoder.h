#pragma once

#include "model.h"
#include "matrix.h"
#include "gru.h"
#include "array.h"

namespace amunmt {

class God;

namespace FPGA {

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
    : w_(initModel)
    , gru_(openCLInfo, gruModel)
    , Temp1_(openCLInfo)
    , Temp2_(openCLInfo)
    {}

    void InitializeState(mblas::Matrix& State,
                         const mblas::Matrix& SourceContext,
                         const size_t batchSize,
                         const Array<int>& mapping)
    {
      using namespace mblas;

      std::cerr << "1State=" << State.Debug(1) << std::endl;
      std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
      Temp2_.Resize(1, SourceContext.dim(1), 1, batchSize);
      std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

      std::cerr << "SourceContext=" << SourceContext.Debug(1) << std::endl;
      std::cerr << "mapping=" << mapping.Debug() << std::endl;
      Mean(Temp2_, SourceContext, mapping);
      std::cerr << "3Temp2_=" << Temp2_.Debug(1) << std::endl;

    }

    private:
      const Weights1& w_;
      const GRU<Weights2> gru_;

      mblas::Matrix Temp1_;
      mblas::Matrix Temp2_;
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

    void Init(const mblas::Matrix& SourceContext)
    {

    }

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
