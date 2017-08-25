#pragma once

#include "gpu/mblas/matrix_functions.h"
#include "model.h"
#include "gru.h"
#include "gpu/types-gpu.h"
#include "common/god.h"

namespace amunmt {
namespace GPU {

class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Rows, const std::vector<size_t>& ids) {
          using namespace mblas;
          HostVector<uint> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_->dim(0))
              id = 1;
          indices_.resize(tids.size());

          mblas::copy(thrust::raw_pointer_cast(tids.data()),
              tids.size(),
              thrust::raw_pointer_cast(indices_.data()),
              cudaMemcpyHostToDevice);

          Assemble(Rows, *w_.E_, indices_);
        }

        size_t GetCols() {
          return w_.E_->dim(1);
        }

        size_t GetRows() const {
          return w_.E_->dim(0);
        }

      private:
        const Weights& w_;
        DeviceVector<uint> indices_;

        Embeddings(const Embeddings&) = delete;
    };

    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel)
        , gru_(gruModel)
        {}

        void InitializeState(mblas::Matrix& State,
                             const mblas::Matrix& SourceContext,
                             const size_t batchSize,
                             const mblas::IMatrix &sentencesMask)
        {
          using namespace mblas;

          //std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
          Temp2_.NewSize(batchSize, SourceContext.dim(1), 1, 1);
          //std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

          //std::cerr << "SourceContext=" << SourceContext.Debug(1) << std::endl;
          //std::cerr << "mapping=" << Debug(mapping, 2) << std::endl;
          Mean(Temp2_, SourceContext, sentencesMask);

          //std::cerr << "1State=" << State.Debug(1) << std::endl;
          //std::cerr << "3Temp2_=" << Temp2_.Debug(1) << std::endl;
          //std::cerr << "w_.Wi_=" << w_.Wi_->Debug(1) << std::endl;
          Prod(State, Temp2_, *w_.Wi_);

          //std::cerr << "2State=" << State.Debug(1) << std::endl;
          //State.ReduceDimensions();

          if (w_.Gamma_->size()) {
            Normalization(State, State, *w_.Gamma_, *w_.Bi_, 1e-9);
            Element(Tanh(_1), State);
          } else {
            BroadcastVec(Tanh(_1 + _2), State, *w_.Bi_);
          }
          //std::cerr << "3State=" << State.Debug(1) << std::endl;
          //std::cerr << "\n";
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

        RNNHidden(const RNNHidden&) = delete;
    };

    template <class Weights>
    class RNNFinal {
      public:
        RNNFinal(const Weights& model)
        : gru_(model) {}

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) {
          gru_.GetNextState(NextState, State, Context);
        }

      private:
        const GRU<Weights> gru_;

        RNNFinal(const RNNFinal&) = delete;
    };

    template <class Weights>
    class Alignment {
      public:
        Alignment(const God &god, const Weights& model)
          : w_(model)
          , dBatchMapping_(god.Get<size_t>("mini-batch") * god.Get<size_t>("beam-size"), 0)
        {}

        void Init(const mblas::Matrix& SourceContext) {
          using namespace mblas;

          Prod(/*h_[0],*/ SCU_, SourceContext, *w_.U_);
          //std::cerr << "SCU_=" << SCU_.Debug(1) << std::endl;

          if (w_.Gamma_1_->size()) {
            Normalization(SCU_, SCU_, *w_.Gamma_1_, *w_.B_, 1e-9);
          }
        }

        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& SourceContext,
                                     const mblas::IMatrix &sentencesMask,
                                     const std::vector<uint>& beamSizes)
        {
          // mapping = 1/0 whether each position, in each sentence in the batch is actually a valid word
          // batchMapping = which sentence is each element in the batch. eg 0 0 1 2 2 2 = first 2 belongs to sent0, 3rd is sent1, 4th and 5th is sent2
          // dBatchMapping_ = fixed length (batch*beam) version of dBatchMapping_

          using namespace mblas;

          size_t batchSize = SourceContext.dim(3);
          //std::cerr << "batchSize=" << batchSize << std::endl;
          //std::cerr << "HiddenState=" << HiddenState.Debug(0) << std::endl;

          HostVector<uint> batchMapping(HiddenState.dim(0));
          size_t k = 0;
          for (size_t i = 0; i < beamSizes.size(); ++i) {
            for (size_t j = 0; j < beamSizes[i]; ++j) {
              batchMapping[k++] = i;
            }
          }

          dBatchMapping_.resize(batchMapping.size());
          mblas::copy(thrust::raw_pointer_cast(batchMapping.data()),
              batchMapping.size(),
              thrust::raw_pointer_cast(dBatchMapping_.data()),
              cudaMemcpyHostToDevice);
          //std::cerr << "mapping=" << Debug(mapping, 2) << std::endl;
          //std::cerr << "batchMapping=" << Debug(batchMapping, 2) << std::endl;
          //std::cerr << "dBatchMapping_=" << Debug(dBatchMapping_, 2) << std::endl;

          const size_t srcSize = sentencesMask.size() / beamSizes.size();

          Prod(/*h_[1],*/ Temp2_, HiddenState, *w_.W_);
          //std::cerr << "1Temp2_=" << Temp2_.Debug() << std::endl;

          if (w_.Gamma_2_->size()) {
            Normalization(Temp2_, Temp2_, *w_.Gamma_2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, Temp2_, *w_.B_/*, s_[1]*/);
          }
          //std::cerr << "2Temp2_=" << Temp2_.Debug() << std::endl;

          Copy(Temp1_, SCU_);
          //std::cerr << "1Temp1_=" << Temp1_.Debug() << std::endl;

          Broadcast(Tanh(_1 + _2), Temp1_, Temp2_, dBatchMapping_, srcSize);

          //std::cerr << "w_.V_=" << w_.V_->Debug(0) << std::endl;
          //std::cerr << "3Temp1_=" << Temp1_.Debug(0) << std::endl;

          Prod(A_, *w_.V_, Temp1_, false, true);

          mblas::Softmax(A_, dBatchMapping_, sentencesMask, batchSize);
          mblas::WeightedMean(AlignedSourceContext, A_, SourceContext, dBatchMapping_);

          /*
          std::cerr << "AlignedSourceContext=" << AlignedSourceContext.Debug() << std::endl;
          std::cerr << "A_=" << A_.Debug() << std::endl;
          std::cerr << "SourceContext=" << SourceContext.Debug() << std::endl;
          std::cerr << "mapping=" << Debug(mapping, 2) << std::endl;
          std::cerr << "dBatchMapping_=" << Debug(dBatchMapping_, 2) << std::endl;
          std::cerr << std::endl;
          */
        }

        void GetAttention(mblas::Matrix& Attention) {
          mblas::Copy(Attention, A_);
        }

        mblas::Matrix& GetAttention() {
          return A_;
        }

      private:
        const Weights& w_;

        DeviceVector<uint> dBatchMapping_;

        mblas::Matrix SCU_;
        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;

        mblas::Matrix Ones_;
        mblas::Matrix Sums_;

        Alignment(const Alignment&) = delete;
    };

    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          mblas::Transpose(TempW4, *w_.W4_);
          mblas::Transpose(TempB4, *w_.B4_);
        }

        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
          using namespace mblas;

          //BEGIN_TIMER("GetProbs.Prod");
          Prod(/*h_[0],*/ T1_, State, *w_.W1_);
          //PAUSE_TIMER("GetProbs.Prod");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec");
          if (w_.Gamma_1_->size()) {
            Normalization(T1_, T1_, *w_.Gamma_1_, *w_.B1_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T1_, *w_.B1_ /*,s_[0]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec");

          //BEGIN_TIMER("GetProbs.Prod2");
          Prod(/*h_[1],*/ T2_, Embedding, *w_.W2_);
          //PAUSE_TIMER("GetProbs.Prod2");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec2");
          if (w_.Gamma_0_->size()) {
            Normalization(T2_, T2_, *w_.Gamma_0_, *w_.B2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T2_, *w_.B2_ /*,s_[1]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec2");

          //BEGIN_TIMER("GetProbs.Prod3");
          Prod(/*h_[2],*/ T3_, AlignedSourceContext, *w_.W3_);
          //PAUSE_TIMER("GetProbs.Prod3");

          //BEGIN_TIMER("GetProbs.Normalization/BroadcastVec3");
          if (w_.Gamma_2_->size()) {
            Normalization(T3_, T3_, *w_.Gamma_2_, *w_.B3_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T3_, *w_.B3_ /*,s_[2]*/);
          }
          //PAUSE_TIMER("GetProbs.Normalization/BroadcastVec3");

          //BEGIN_TIMER("GetProbs.Element");
          Element(Tanh(_1 + _2 + _3), T1_, T2_, T3_);
          //PAUSE_TIMER("GetProbs.Element");

          std::shared_ptr<mblas::Matrix> w4, b4;
          if(!filtered_) {
            w4 = w_.W4_;
            b4 = w_.B4_;
          } else {
            w4.reset(&FilteredW4_);
            b4.reset(&FilteredB4_);
          }

          //BEGIN_TIMER("GetProbs.NewSize");
          Probs.NewSize(T1_.dim(0), w4->dim(1));
          //PAUSE_TIMER("GetProbs.NewSize");

          //BEGIN_TIMER("GetProbs.Prod4");
          Prod(Probs, T1_, *w4);
          //PAUSE_TIMER("GetProbs.Prod4");

          //BEGIN_TIMER("GetProbs.BroadcastVec");
          BroadcastVec(_1 + _2, Probs, *b4);
          //PAUSE_TIMER("GetProbs.BroadcastVec");

          //BEGIN_TIMER("GetProbs.LogSoftMax");
          mblas::LogSoftmax(Probs);
          //PAUSE_TIMER("GetProbs.LogSoftMax");
        }

        void Filter(const std::vector<size_t>& ids) {
          filtered_ = true;
          using namespace mblas;

          Assemble(FilteredW4_, TempW4, ids);
          Assemble(FilteredB4_, TempB4, ids);

          Transpose(FilteredW4_);
          Transpose(FilteredB4_);
        }

      private:
        const Weights& w_;

        bool filtered_;
        mblas::Matrix FilteredW4_;
        mblas::Matrix FilteredB4_;

        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;

        mblas::Matrix TempW4;
        mblas::Matrix TempB4;

        Softmax(const Softmax&) = delete;
    };

  public:
    Decoder(const God &god, const Weights& model)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, model.decGru1_),
      rnn2_(model.decGru2_),
      alignment_(god, model.decAlignment_),
      softmax_(model.decSoftmax_)
    {}

    void Decode(mblas::Matrix& NextState,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embeddings,
                  const mblas::Matrix& SourceContext,
                  const mblas::IMatrix &sentencesMask,
                  const std::vector<uint>& beamSizes)
    {
      //BEGIN_TIMER("Decode");

      //BEGIN_TIMER("GetHiddenState");
      //std::cerr << "State=" << State.Debug(1) << std::endl;
      //std::cerr << "Embeddings=" << Embeddings.Debug(1) << std::endl;
      GetHiddenState(HiddenState_, State, Embeddings);
      //HiddenState_.ReduceDimensions();
      //std::cerr << "HiddenState_=" << HiddenState_.Debug(1) << std::endl;
      //PAUSE_TIMER("GetHiddenState");

      //BEGIN_TIMER("GetAlignedSourceContext");
      GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext, sentencesMask, beamSizes);
      //std::cerr << "AlignedSourceContext_=" << AlignedSourceContext_.Debug(1) << std::endl;
      //PAUSE_TIMER("GetAlignedSourceContext");

      //BEGIN_TIMER("GetNextState");
      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      //std::cerr << "NextState=" << NextState.Debug(1) << std::endl;
      //PAUSE_TIMER("GetNextState");

      //BEGIN_TIMER("GetProbs");
      GetProbs(NextState, Embeddings, AlignedSourceContext_);
      //std::cerr << "Probs_=" << Probs_.Debug(1) << std::endl;
      //PAUSE_TIMER("GetProbs");

      //PAUSE_TIMER("Decode");
    }

    mblas::Matrix& GetProbs() {
      return Probs_;
    }

    void EmptyState(mblas::Matrix& State,
                    const mblas::Matrix& SourceContext,
                    size_t batchSize,
                    const mblas::IMatrix &sentencesMask)
    {
      rnn1_.InitializeState(State, SourceContext, batchSize, sentencesMask);
      alignment_.Init(SourceContext);
    }

    void EmptyEmbedding(mblas::Matrix& Embedding, size_t batchSize = 1) {
      Embedding.NewSize(batchSize, embeddings_.GetCols());
      mblas::Fill(Embedding, 0);
    }

    void Lookup(mblas::Matrix& Embedding,
                const std::vector<size_t>& w) {
      embeddings_.Lookup(Embedding, w);
    }

    void Filter(const std::vector<size_t>& ids) {
      softmax_.Filter(ids);
    }

    void GetAttention(mblas::Matrix& Attention) {
      alignment_.GetAttention(Attention);
    }

    size_t GetVocabSize() const {
      return embeddings_.GetRows();
    }

    mblas::Matrix& GetAttention() {
      return alignment_.GetAttention();
    }

  private:

    void GetHiddenState(mblas::Matrix& HiddenState,
                        const mblas::Matrix& PrevState,
                        const mblas::Matrix& Embedding) {
      rnn1_.GetNextState(HiddenState, PrevState, Embedding);
    }

    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                  const mblas::Matrix& HiddenState,
                                  const mblas::Matrix& SourceContext,
                                  const mblas::IMatrix &sentencesMask,
                                  const std::vector<uint>& beamSizes) {
      alignment_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext,
                                         sentencesMask, beamSizes);
    }

    void GetNextState(mblas::Matrix& State,
                      const mblas::Matrix& HiddenState,
                      const mblas::Matrix& AlignedSourceContext) {
      rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
    }


    void GetProbs(const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
      softmax_.GetProbs(Probs_, State, Embedding, AlignedSourceContext);
    }

  private:
    mblas::Matrix HiddenState_;
    mblas::Matrix AlignedSourceContext_;
    mblas::Matrix Probs_;

    Embeddings<Weights::DecEmbeddings> embeddings_;
    RNNHidden<Weights::DecInit, Weights::DecGRU1> rnn1_;
    RNNFinal<Weights::DecGRU2> rnn2_;
    Alignment<Weights::DecAlignment> alignment_;
    Softmax<Weights::DecSoftmax> softmax_;

    Decoder(const Decoder&) = delete;
};

}
}

