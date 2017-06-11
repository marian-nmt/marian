#pragma once

#include "model.h"
#include "matrix.h"
#include "gru.h"
#include "array.h"
#include "common/god.h"

namespace amunmt {

class God;

namespace FPGA {

class Decoder {

  template <class Weights>
  class Embeddings {
  public:
    Embeddings(const OpenCLInfo &openCLInfo, const Weights& model)
    : w_(model)
    , indices_(openCLInfo)
    {}

    void Lookup(mblas::Matrix& Rows, const std::vector<uint>& ids)
    {
      using namespace mblas;
      std::vector<uint> tids = ids;
      for(auto&& id : tids)
        if(id >= w_.E_.dim(0))
          id = 1;
      indices_.resize(tids.size());

      indices_.Set(tids);

      Assemble(Rows, w_.E_, indices_);
    }

    size_t GetCols() {
      return w_.E_.dim(1);
    }

    size_t GetRows() const {
      return w_.E_.dim(0);
    }

  private:
    const Weights& w_;
    Array<uint> indices_;

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

      //std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
      Temp2_.Resize(1, SourceContext.dim(1), 1, batchSize);
      //std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

      //std::cerr << "SourceContext=" << SourceContext.Debug(1) << std::endl;
      //std::cerr << "mapping=" << mapping.Debug() << std::endl;
      Mean(Temp2_, SourceContext, mapping);

      //std::cerr << "1State=" << State.Debug(1) << std::endl;
      //std::cerr << "3Temp2_=" << Temp2_.Debug(1) << std::endl;
      //std::cerr << "w_.Wi_=" << w_.Wi_.Debug(1) << std::endl;
      Prod(State, Temp2_, w_.Wi_);
      //std::cerr << "2State=" << State.Debug(1) << std::endl;

      if (w_.Gamma_.size()) {
        //TODO
        //Normalization(State, State, w_.Gamma_, w_.Bi_, 1e-9);
      }
      else {
        BroadcastVecTanh(State, w_.Bi_);
      }
      //std::cerr << "3State=" << State.Debug(1) << std::endl;

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

    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context)
    {
      gru_.GetNextState(NextState, State, Context);
    }

    private:
      const GRU<Weights> gru_;
  };

  template <class Weights>
  class Alignment {
    public:
    Alignment(const OpenCLInfo &openCLInfo, const God &god, const Weights& model)
      : w_(model)
      , dBatchMapping_(openCLInfo, god.Get<size_t>("mini-batch") * god.Get<size_t>("beam-size"), 0)
      , SCU_(openCLInfo)
      , Temp1_(openCLInfo)
      , Temp2_(openCLInfo)
      , A_(openCLInfo)
    {
    }

    void Init(const mblas::Matrix& SourceContext)
    {
      using namespace mblas;

      Prod(/*h_[0],*/ SCU_, SourceContext, w_.U_);
      //std::cerr << "SCU_=" << SCU_.Debug(1) << std::endl;

      // TODO
      if (w_.Gamma_1_.size()) {
        //Normalization(SCU_, SCU_, w_.Gamma_1_, w_.B_, 1e-9);
      }
    }

    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                 const mblas::Matrix& HiddenState,
                                 const mblas::Matrix& SourceContext,
                                 const Array<int>& mapping,
                                 const std::vector<uint>& beamSizes)
    {
      // mapping = 1/0 whether each position, in each sentence in the batch is actually a valid word
      // batchMapping = which sentence is each element in the batch. eg 0 0 1 2 2 2 = first 2 belongs to sent0, 3rd is sent1, 4th and 5th is sent2
      // dBatchMapping_ = fixed length (batch*beam) version of dBatchMapping_

      using namespace mblas;
      const OpenCLInfo &openCLInfo = HiddenState.GetOpenCLInfo();

      std::vector<int> batchMapping(HiddenState.dim(0));
      size_t k = 0;
      for (size_t i = 0; i < beamSizes.size(); ++i) {
        for (size_t j = 0; j < beamSizes[i]; ++j) {
          batchMapping[k++] = i;
        }
      }

      //std::cerr << "batchMapping=" << Debug(batchMapping) << std::endl;
      dBatchMapping_.Set(batchMapping);
      //std::cerr << "mapping=" << mapping.Debug(2) << std::endl;
      //std::cerr << "batchMapping=" << Debug(batchMapping, 2) << std::endl;
      //std::cerr << "dBatchMapping_=" << dBatchMapping_.Debug(2) << std::endl;

      const size_t srcSize = mapping.size() / beamSizes.size();

      Prod(/*h_[1],*/ Temp2_, HiddenState, w_.W_);
      //std::cerr << "1Temp2_=" << Temp2_.Debug() << std::endl;

      if (w_.Gamma_2_.size()) {
        //Normalization(Temp2_, Temp2_, w_.Gamma_2_, 1e-9);
      } else {
        BroadcastVecAdd(Temp2_, w_.B_/*, s_[1]*/);
      }
      //std::cerr << "2Temp2_=" << Temp2_.Debug() << std::endl;

      Copy(Temp1_, SCU_);
      //std::cerr << "1Temp1_=" << Temp1_.Debug() << std::endl;

      BroadcastTanh(Temp1_, Temp2_, dBatchMapping_, srcSize);
      //std::cerr << "2Temp1_=" << Temp1_.Debug() << std::endl;

      Temp1_.Reshape2D();

      Transpose(Temp1_);

      //std::cerr << "w_.V_=" << w_.V_.Debug() << std::endl;
      //std::cerr << "3Temp1_=" << Temp1_.Debug() << std::endl;
      Prod(A_, w_.V_, Temp1_, false, false);

      size_t rows1 = SourceContext.dim(0);
      size_t rows2 = HiddenState.dim(0);

      A_.Reshape(rows2, srcSize, 1, 1); // due to broadcasting above
      mblas::Softmax(A_, dBatchMapping_, mapping, srcSize);

      AlignedSourceContext.Resize(A_.dim(0), SourceContext.dim(1));

      //std::cerr << "1AlignedSourceContext=" << AlignedSourceContext.Debug() << std::endl;
      //std::cerr << "A_=" << A_.Debug() << std::endl;
      //std::cerr << "SourceContext=" << SourceContext.Debug() << std::endl;
      //std::cerr << "dBatchMapping_=" << dBatchMapping_.Debug(2) << std::endl;
      mblas::WeightedMean(AlignedSourceContext, A_, SourceContext, dBatchMapping_);
      //std::cerr << "2AlignedSourceContext=" << AlignedSourceContext.Debug() << std::endl;
    }

    mblas::Matrix& GetAttention() {
      return A_;
    }

    private:
      const Weights& w_;

      Array<int> dBatchMapping_;

      mblas::Matrix SCU_;
      mblas::Matrix Temp1_;
      mblas::Matrix Temp2_;
      mblas::Matrix A_;

  };

  template <class Weights>
  class Softmax {
  public:
    Softmax(const OpenCLInfo &openCLInfo, const Weights& model)
    : w_(model), filtered_(false)
    , T1_(openCLInfo)
    , T2_(openCLInfo)
    , T3_(openCLInfo)
    {
    }

    void GetProbs(mblas::Matrix& Probs,
              const mblas::Matrix& State,
              const mblas::Matrix& Embedding,
              const mblas::Matrix& AlignedSourceContext)
    {
      using namespace mblas;

      Prod(/*h_[0],*/ T1_, State, w_.W1_);

      if (w_.Gamma_1_.size()) {
        //Normalization(T1_, T1_, w_.Gamma_1_, w_.B1_, 1e-9);
      } else {
        BroadcastVecAdd(T1_, w_.B1_ /*,s_[0]*/);
      }

      Prod(/*h_[1],*/ T2_, Embedding, w_.W2_);

      if (w_.Gamma_0_.size()) {
        //Normalization(T2_, T2_, w_.Gamma_0_, w_.B2_, 1e-9);
      } else {
        BroadcastVecAdd(T2_, w_.B2_ /*,s_[1]*/);
      }

      Prod(/*h_[2],*/ T3_, AlignedSourceContext, w_.W3_);

      if (w_.Gamma_2_.size()) {
        //Normalization(T3_, T3_, w_.Gamma_2_, w_.B3_, 1e-9);
      } else {
        BroadcastVecAdd(T3_, w_.B3_ /*,s_[2]*/);
      }

      ElementTanh2(T1_, T2_, T3_);

      if(!filtered_) {
        Probs.Resize(T1_.dim(0), w_.W4_.dim(1));
        Prod(Probs, T1_, w_.W4_);
        BroadcastVecAdd(Probs, w_.B4_);
      } else {
        //Probs.Resize(T1_.dim(0), FilteredW4_.dim(1));
        //Prod(Probs, T1_, FilteredW4_);
        //BroadcastVec(_1 + _2, Probs, FilteredB4_);
      }

      mblas::LogSoftmax(Probs);
    }

  private:
    const Weights& w_;
    bool filtered_;

    mblas::Matrix T1_;
    mblas::Matrix T2_;
    mblas::Matrix T3_;

  };

public:
  Decoder(const OpenCLInfo &openCLInfo, const God &god, const Weights& model)
  : HiddenState_(openCLInfo),
    AlignedSourceContext_(openCLInfo),
    Probs_(openCLInfo),
    embeddings_(openCLInfo, model.decEmbeddings_),
    rnn1_(openCLInfo, model.decInit_, model.decGru1_),
    rnn2_(openCLInfo, model.decGru2_),
    alignment_(openCLInfo, god, model.decAlignment_),
    softmax_(openCLInfo, model.decSoftmax_)
  {}

  size_t GetVocabSize() const {
    return embeddings_.GetRows();
  }

  mblas::Matrix& GetProbs() {
    return Probs_;
  }

  mblas::Matrix& GetAttention() {
    return alignment_.GetAttention();
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
                const std::vector<uint>& beamSizes);

  void GetHiddenState(mblas::Matrix& HiddenState,
                      const mblas::Matrix& PrevState,
                      const mblas::Matrix& Embedding);

  void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                const mblas::Matrix& HiddenState,
                                const mblas::Matrix& SourceContext,
                                const Array<int>& mapping,
                                const std::vector<uint>& beamSizes);

  void GetNextState(mblas::Matrix& State,
                    const mblas::Matrix& HiddenState,
                    const mblas::Matrix& AlignedSourceContext);

  void GetProbs(const mblas::Matrix& State,
                const mblas::Matrix& Embedding,
                const mblas::Matrix& AlignedSourceContext);

  void Lookup(mblas::Matrix& Embedding,
              const std::vector<uint>& w);

private:
  mblas::Matrix HiddenState_;
  mblas::Matrix AlignedSourceContext_;
  mblas::Matrix Probs_;

  Embeddings<Weights::DecEmbeddings> embeddings_;
  RNNHidden<Weights::DecInit, Weights::DecGRU1> rnn1_;
  RNNFinal<Weights::DecGRU2> rnn2_;
  Alignment<Weights::DecAlignment> alignment_;
  Softmax<Weights::DecSoftmax> softmax_;

};

}
}
