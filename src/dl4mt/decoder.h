#pragma once

#include "mblas/matrix.h"
#include "dl4mt/model.h"
#include "dl4mt/gru.h"

class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Rows, const std::vector<size_t>& ids) {
          std::vector<size_t> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_.Rows())
              id = 1;
          mblas::Assemble(Rows, w_.E_, tids);
        }

        size_t GetCols() {
          return w_.E_.Cols();
        }

        size_t GetRows() const {
          return w_.E_.Rows();
        }

      private:
        const Weights& w_;
    };

    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel), gru_(gruModel) {}

        void InitializeState(mblas::Matrix& State,
                             const std::vector<mblas::Matrix>& SourceContexts,
                             const size_t batchSize = 1) {
          ConcatenateSourceContexts(SourceContexts);
          Temp2_.Clear();
          Temp2_.Resize(batchSize, ConcatenatedSourceContext_.Cols(), 0.0);
          mblas::BroadcastVec(mblas::_1 + mblas::_2, Temp2_, Temp1_);
          mblas::Prod(State, Temp2_, w_.Wi_);
          mblas::BroadcastVec(Tanh(mblas::_1 + mblas::_2), State, w_.Bi_);
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Context) {
          gru_.GetNextState(NextState, State, Context);
        }

      private:
        void ConcatenateSourceContexts(const std::vector<mblas::Matrix>& SourceContexts) {
          ConcatenatedSourceContext_.Resize(1, 0);
          for (const auto& sourceContext : SourceContexts) {
            mblas::Mean(Temp1_, sourceContext);
            mblas::ConcatRows(ConcatenatedSourceContext_, Temp1_);
          }
        }
        const Weights1& w_;
        const GRU<Weights2> gru_;

        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix ConcatenatedSourceContext_;
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
    };

    template <class Weights>
    class Alignment {
      public:
        Alignment(const Weights& model)
        : w_(model)
        {
          //for(int i = 0; i < 2; ++i) {
          //  cudaStreamCreate(&s_[i]);
          //  cublasCreate(&h_[i]);
          //  cublasSetStream(h_[i], s_[i]);
          //}
        }

        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& SourceContext) {
          using namespace mblas;

          Prod(/*h_[0],*/ Temp1_, SourceContext, w_.U_);
          Prod(/*h_[1],*/ Temp2_, HiddenState, w_.W_);
          BroadcastVec(_1 + _2, Temp2_, w_.B_/*, s_[1]*/);

          //cudaDeviceSynchronize();

          Broadcast(Tanh(_1 + _2), Temp1_, Temp2_);

          Prod(A_, w_.V_, Temp1_, false, true);

          size_t rows1 = SourceContext.Rows();
          size_t rows2 = HiddenState.Rows();
          A_.Reshape(rows2, rows1); // due to broadcasting above
          Element(_1 + w_.C_(0,0), A_);

          mblas::Softmax(A_);
          Prod(AlignedSourceContext, A_, SourceContext);
        }

        void GetAttention(mblas::Matrix& Attention) {
          mblas::Copy(Attention, A_);
        }

      private:
        const Weights& w_;

        cublasHandle_t h_[2];
        cudaStream_t s_[2];

        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;

        mblas::Matrix Ones_;
        mblas::Matrix Sums_;
    };

    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model), filtered_(false)
        {
          //for(int i = 0; i < 3; ++i) {
          //  cudaStreamCreate(&s_[i]);
          //  cublasCreate(&h_[i]);
          //  cublasSetStream(h_[i], s_[i]);
          //}
        }

        void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
          using namespace mblas;

          Prod(/*h_[1],*/ T2_, Embedding, w_.W2_);
          Prod(/*h_[2],*/ T3_, AlignedSourceContext, w_.W3_);
          Prod(/*h_[0],*/ T1_, State, w_.W1_);

          BroadcastVec(_1 + _2, T1_, w_.B1_ /*,s_[0]*/);
          BroadcastVec(_1 + _2, T2_, w_.B2_ /*,s_[1]*/);
          BroadcastVec(_1 + _2, T3_, w_.B3_ /*,s_[2]*/);

          //cudaDeviceSynchronize();

          Element(Tanh(_1 + _2 + _3), T1_, T2_, T3_);

          Prod(Probs, T1_, w_.W4_);
          BroadcastVec(_1 + _2, Probs, w_.B4_);
          mblas::Softmax(Probs);
          Element(Log(_1), Probs);
        }

        void Filter(const std::vector<size_t>& ids) {
        }

      private:
        const Weights& w_;

        cublasHandle_t h_[3];
        cudaStream_t s_[3];

        bool filtered_;
        mblas::Matrix FilteredWo_;
        mblas::Matrix FilteredWoB_;

        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;
    };

  public:
    Decoder(const Weights& model, size_t numEncoders)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, model.decGru1_),
      rnn2_(model.decGru2_),
      softmax_(model.decSoftmax_),
      AlignedSourceContexts_(numEncoders)
    {
      for(size_t i = 0; i < numEncoders; ++i) {
        alignments_.emplace_back(*model.decAlignment_[i]);
      }
    }

    void MakeStep(mblas::Matrix& NextState,
                  mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embeddings,
                  const std::vector<mblas::Matrix>& SourceContexts) {
      GetHiddenState(HiddenState_, State, Embeddings);
      GetAlignedSourceContext(AlignedSourceContexts_, HiddenState_, SourceContexts);
      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      GetProbs(Probs, NextState, Embeddings, AlignedSourceContext_);
    }

    void EmptyState(mblas::Matrix& State,
                    const std::vector<mblas::Matrix>& SourceContexts,
                    size_t batchSize = 1) {
      rnn1_.InitializeState(State, SourceContexts, batchSize);
    }

    void EmptyEmbedding(mblas::Matrix& Embedding,
                        size_t batchSize = 1) {
      Embedding.Clear();
      Embedding.Resize(batchSize, embeddings_.GetCols(), 0);
    }

    void Lookup(mblas::Matrix& Embedding,
                const std::vector<size_t>& w) {
      embeddings_.Lookup(Embedding, w);
    }

    void Filter(const std::vector<size_t>& ids) {
      // @TODO
    }

    void GetAttention(mblas::Matrix& Attention, size_t index) {
      alignments_[index].GetAttention(Attention);
    }

    size_t GetVocabSize() const {
      return embeddings_.GetRows();
    }

  private:

    void GetHiddenState(mblas::Matrix& HiddenState,
                        const mblas::Matrix& PrevState,
                        const mblas::Matrix& Embedding) {
      rnn1_.GetNextState(HiddenState, PrevState, Embedding);
    }

    void GetAlignedSourceContext(std::vector<mblas::Matrix>& AlignedSourceContexts,
                                 const mblas::Matrix& HiddenState,
                                 const std::vector<mblas::Matrix>& SourceContext) {
      for (size_t i = 0; i < alignments_.size(); ++i) {
        alignments_[i].GetAlignedSourceContext(AlignedSourceContexts[i],
                                              HiddenState,
                                              SourceContext[i]);
      }

      AlignedSourceContext_.Resize(0, HiddenState.Rows());

      for (auto& sourceContext : AlignedSourceContexts_) {
        mblas::Transpose(sourceContext);
        mblas::Concat(AlignedSourceContext_, sourceContext);
      }
      mblas::Transpose(AlignedSourceContext_);
    }

    void GetNextState(mblas::Matrix& State,
                      const mblas::Matrix& HiddenState,
                      const mblas::Matrix& AlignedSourceContext) {
      rnn2_.GetNextState(State, HiddenState, AlignedSourceContext);
    }


    void GetProbs(mblas::Matrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
      softmax_.GetProbs(Probs, State, Embedding, AlignedSourceContext);
    }

  private:
    mblas::Matrix HiddenState_;
    mblas::Matrix AlignedSourceContext_;
    std::vector<mblas::Matrix> AlignedSourceContexts_;

    Embeddings<Weights::DecEmbeddings> embeddings_;
    RNNHidden<Weights::DecInit, Weights::DecGRU1> rnn1_;
    RNNFinal<Weights::DecGRU2> rnn2_;
    std::vector<Alignment<Weights::DecAlignment>> alignments_;
    Softmax<Weights::DecSoftmax> softmax_;
};
