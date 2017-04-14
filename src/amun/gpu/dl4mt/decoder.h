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
          thrust::host_vector<size_t> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_.dim(0))
              id = 1;
          indices_.resize(tids.size());

          mblas::copy(thrust::raw_pointer_cast(tids.data()),
              tids.size(),
              thrust::raw_pointer_cast(indices_.data()),
              cudaMemcpyHostToDevice);

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
        DeviceVector<size_t> indices_;

        Embeddings(const Embeddings&) = delete;
    };

    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel), gru_(gruModel) {}

        void InitializeState(mblas::Matrix& State,
                             const mblas::Matrix& SourceContext,
                             const size_t batchSize,
                             const DeviceVector<int>& mapping) {
          using namespace mblas;

          //std::cerr << "1State=" << State.Debug(1) << std::endl;
          //std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
          Temp2_.Resize(1, SourceContext.dim(1), 1, batchSize);
          //std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

          //std::cerr << "SourceContext=" << SourceContext.Debug(1) << std::endl;
          //std::cerr << "mapping=" << Debug(mapping) << std::endl;
          Mean(Temp2_, SourceContext, mapping);
          //std::cerr << "3Temp2_=" << Temp2_.Debug(1) << std::endl;

          Prod(State, Temp2_, w_.Wi_);
          //std::cerr << "State=" << State.Debug(1) << std::endl;

          if (w_.Gamma_) {
            Normalization(State, State, w_.Gamma_, w_.Bi_, 1e-9);
          } else {
            BroadcastVec(Tanh(_1 + _2), State, w_.Bi_);
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

          Prod(/*h_[0],*/ SCU_, SourceContext, w_.U_);
          //std::cerr << "SCU_=" << SCU_.Debug(1) << std::endl;

          if (w_.Gamma_1_) {
            Normalization(SCU_, SCU_, w_.Gamma_1_, w_.B_, 1e-9);
          }
        }

        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& SourceContext,
                                     const DeviceVector<int>& mapping,
                                     const std::vector<size_t>& beamSizes) {
          using namespace mblas;

          thrust::host_vector<int> batchMapping(HiddenState.dim(0));
          size_t k = 0;
          for (size_t i = 0; i < beamSizes.size(); ++i) {
            for (size_t j = 0; j < beamSizes[i]; ++j) {
              batchMapping[k++] = i;
            }
          }
          //std::cerr << "batchMapping=" << Debug(batchMapping) << std::endl;

          mblas::copy(thrust::raw_pointer_cast(batchMapping.data()),
              batchMapping.size(),
              thrust::raw_pointer_cast(dBatchMapping_.data()),
              cudaMemcpyHostToDevice);
          //std::cerr << "dBatchMapping_=" << Debug(dBatchMapping_) << std::endl;

          const size_t srcSize = mapping.size() / beamSizes.size();

          Prod(/*h_[1],*/ Temp2_, HiddenState, w_.W_);

          if (w_.Gamma_2_) {
            Normalization(Temp2_, Temp2_, w_.Gamma_2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, Temp2_, w_.B_/*, s_[1]*/);
          }

          Copy(Temp1_, SCU_);

          Broadcast(Tanh(_1 + _2), Temp1_, Temp2_, dBatchMapping_, srcSize);

          Temp1_.Reshape2D();

          Prod(A_, w_.V_, Temp1_, false, true);

          size_t rows1 = SourceContext.dim(0);
          size_t rows2 = HiddenState.dim(0);

          //std::cerr << "1A_=" << A_.Debug() << std::endl;
          A_.Reshape(rows2, srcSize, 1, 1); // due to broadcasting above

          mblas::Softmax(A_, dBatchMapping_, mapping, srcSize);

          AlignedSourceContext.Resize(A_.dim(0), SourceContext.dim(1));

          mblas::WeightedMean(AlignedSourceContext, A_, SourceContext, dBatchMapping_);
        }

        void GetAttention(mblas::Matrix& Attention) {
          mblas::Copy(Attention, A_);
        }

        mblas::Matrix& GetAttention() {
          return A_;
        }

      private:
        const Weights& w_;

        cublasHandle_t h_[2];
        cudaStream_t s_[2];

        DeviceVector<int> dBatchMapping_;

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
          mblas::Transpose(TempW4, w_.W4_);
          mblas::Transpose(TempB4, w_.B4_);
        }

        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
          using namespace mblas;

          Prod(/*h_[0],*/ T1_, State, w_.W1_);

          if (w_.Gamma_1_) {
            Normalization(T1_, T1_, w_.Gamma_1_, w_.B1_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T1_, w_.B1_ /*,s_[0]*/);
          }

          Prod(/*h_[1],*/ T2_, Embedding, w_.W2_);

          if (w_.Gamma_0_) {
            Normalization(T2_, T2_, w_.Gamma_0_, w_.B2_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T2_, w_.B2_ /*,s_[1]*/);
          }

          Prod(/*h_[2],*/ T3_, AlignedSourceContext, w_.W3_);

          if (w_.Gamma_2_) {
            Normalization(T3_, T3_, w_.Gamma_2_, w_.B3_, 1e-9);
          } else {
            BroadcastVec(_1 + _2, T3_, w_.B3_ /*,s_[2]*/);
          }

          Element(Tanh(_1 + _2 + _3), T1_, T2_, T3_);

          if(!filtered_) {
            Probs.Resize(T1_.dim(0), w_.W4_.dim(1));
            Prod(Probs, T1_, w_.W4_);
            BroadcastVec(_1 + _2, Probs, w_.B4_);
          } else {
            Probs.Resize(T1_.dim(0), FilteredW4_.dim(1));
            Prod(Probs, T1_, FilteredW4_);
            BroadcastVec(_1 + _2, Probs, FilteredB4_);
          }

          mblas::LogSoftmax(Probs);
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

        cublasHandle_t h_[3];
        cudaStream_t s_[3];

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
                  const DeviceVector<int>& mapping,
                  const std::vector<size_t>& beamSizes) {
      //std::cerr << std::endl;

      GetHiddenState(HiddenState_, State, Embeddings);
      GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext, mapping, beamSizes);
      std::cerr << "AlignedSourceContext_=" << AlignedSourceContext_.Debug(1) << std::endl;

      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      std::cerr << "NextState=" << NextState.Debug(1) << std::endl;

      GetProbs(NextState, Embeddings, AlignedSourceContext_);
      std::cerr << "Probs_=" << Probs_.Debug(1) << std::endl;
      
    }

    mblas::Matrix& GetProbs() {
      return Probs_;
    }

    void EmptyState(mblas::Matrix& State,
                    const mblas::Matrix& SourceContext,
                    size_t batchSize,
                    const DeviceVector<int>& batchMapping) {
      rnn1_.InitializeState(State, SourceContext, batchSize, batchMapping);
      alignment_.Init(SourceContext);
    }

    void EmptyEmbedding(mblas::Matrix& Embedding, size_t batchSize = 1) {
      Embedding.Clear();
      Embedding.Resize(batchSize, embeddings_.GetCols());
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
                                  const DeviceVector<int>& mapping,
                                  const std::vector<size_t>& beamSizes) {
      alignment_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext,
                                         mapping, beamSizes);
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

