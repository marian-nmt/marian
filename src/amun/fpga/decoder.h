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

    size_t GetCols() {
      return w_.E_.dim(1);
    }

    size_t GetRows() const {
      return w_.E_.dim(0);
    }

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

      //std::cerr << "1State=" << State.Debug(1) << std::endl;
      //std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
      Temp2_.Resize(1, SourceContext.dim(1), 1, batchSize);
      //std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

      //std::cerr << "SourceContext=" << SourceContext.Debug(1) << std::endl;
      //std::cerr << "mapping=" << mapping.Debug() << std::endl;
      Mean(Temp2_, SourceContext, mapping);
      //std::cerr << "3Temp2_=" << Temp2_.Debug(1) << std::endl;

      Prod(State, Temp2_, w_.Wi_);
      //std::cerr << "State=" << State.Debug(1) << std::endl;

      if (w_.Gamma_) {
        //TODO
        //Normalization(State, State, w_.Gamma_, w_.Bi_, 1e-9);
      }
      else {
        BroadcastVecTanh(State, w_.Bi_);
      }

    }

    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) {
      gru_.GetNextState(NextState, State, Context);
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
    Alignment(const OpenCLInfo &openCLInfo, const God &god, const Weights& model)
      : w_(model)
      , SCU_(openCLInfo)
    {}

    void Init(const mblas::Matrix& SourceContext)
    {
      using namespace mblas;

      Prod(/*h_[0],*/ SCU_, SourceContext, w_.U_);
      //std::cerr << "SCU_=" << SCU_.Debug(1) << std::endl;

      // TODO
      if (w_.Gamma_1_) {
        //Normalization(SCU_, SCU_, w_.Gamma_1_, w_.B_, 1e-9);
      }
    }

    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                 const mblas::Matrix& HiddenState,
                                 const mblas::Matrix& SourceContext,
                                 const Array<int>& mapping,
                                 const std::vector<size_t>& beamSizes)
    {
      using namespace mblas;

      std::vector<int> batchMapping(HiddenState.dim(0));
      size_t k = 0;
      for (size_t i = 0; i < beamSizes.size(); ++i) {
        for (size_t j = 0; j < beamSizes[i]; ++j) {
          batchMapping[k++] = i;
        }
      }

    }

    private:
      const Weights& w_;

      mblas::Matrix SCU_;

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
  : HiddenState_(openCLInfo),
    AlignedSourceContext_(openCLInfo),
    embeddings_(model.decEmbeddings_),
    rnn1_(openCLInfo, model.decInit_, model.decGru1_),
    rnn2_(openCLInfo, model.decGru2_),
    alignment_(openCLInfo, god, model.decAlignment_),
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

  void EmptyEmbedding(mblas::Matrix& Embedding, size_t batchSize = 1);

  void Decode(mblas::Matrix& NextState,
                const mblas::Matrix& State,
                const mblas::Matrix& Embeddings,
                const mblas::Matrix& SourceContext,
                const Array<int>& mapping,
                const std::vector<size_t>& beamSizes);

  void GetHiddenState(mblas::Matrix& HiddenState,
                      const mblas::Matrix& PrevState,
                      const mblas::Matrix& Embedding);

  void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                const mblas::Matrix& HiddenState,
                                const mblas::Matrix& SourceContext,
                                const Array<int>& mapping,
                                const std::vector<size_t>& beamSizes);

private:
  mblas::Matrix HiddenState_;
  mblas::Matrix AlignedSourceContext_;

  Embeddings<Weights::DecEmbeddings> embeddings_;
  RNNHidden<Weights::DecInit, Weights::DecGRU1> rnn1_;
  RNNFinal<Weights::DecGRU2> rnn2_;
  Alignment<Weights::DecAlignment> alignment_;
  Softmax<Weights::DecSoftmax> softmax_;

};

}
}
