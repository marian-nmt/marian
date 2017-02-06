#pragma once

#include "../mblas/matrix.h"
#include "model.h"
#include "gru.h"
#include "common/god.h"

namespace amunmt {
namespace CPU {

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
          std::vector<size_t> tids = ids;
          for(auto&& id : tids)
            if(id >= w_.E_.rows())
              id = 1;
          Rows = Assemble<byRow, Matrix>(w_.E_, tids);
        }

        size_t GetCols() {
          return w_.E_.columns();
        }

        size_t GetRows() const {
          return w_.E_.rows();
        }

      private:
        const Weights& w_;
    };

    //////////////////////////////////////////////////////////////
    template <class Weights1, class Weights2>
    class RNNHidden {
      public:
        RNNHidden(const Weights1& initModel, const Weights2& gruModel)
        : w_(initModel), gru_(gruModel) {}

        void InitializeState(mblas::Matrix& State,
                             const mblas::Matrix& SourceContext,
                             const size_t batchSize = 1) {
          using namespace mblas;

          // Calculate mean of source context, rowwise
          // Repeat mean batchSize times by broadcasting
          Temp1_ = Mean<byRow, Matrix>(SourceContext);
          Temp2_.resize(batchSize, SourceContext.columns());
          Temp2_ = 0.0f;
          AddBiasVector<byRow>(Temp2_, Temp1_);

          State = Temp2_ * w_.Wi_;

          AddBiasVector<byRow>(State, w_.Bi_);

          State = blaze::forEach(State, Tanh());
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

    //////////////////////////////////////////////////////////////
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

    //////////////////////////////////////////////////////////////
    template <class Weights>
    class Attention {
      public:
        Attention(const Weights& model)
        : w_(model)
        {
          V_ = blaze::trans(blaze::row(w_.V_, 0));
        }

        void Init(const mblas::Matrix& SourceContext) {
          using namespace mblas;
          SCU_ = SourceContext * w_.U_;
          AddBiasVector<byRow>(SCU_, w_.B_);
        }

        void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                     const mblas::Matrix& HiddenState,
                                     const mblas::Matrix& SourceContext) {
          using namespace mblas;

          Temp2_ = HiddenState * w_.W_;

          // For batching: create an A across different sentences,
          // maybe by mapping and looping. In the and join different
          // alignment matrices into one
          // Or masking?
          Temp1_ = Broadcast<Matrix>(Tanh(), SCU_, Temp2_);

          A_.resize(Temp1_.rows(), 1);
          blaze::column(A_, 0) = Temp1_ * V_;
          size_t words = SourceContext.rows();
          // batch size, for batching, divide by numer of sentences
          size_t batchSize = HiddenState.rows();
          Reshape(A_, batchSize, words); // due to broadcasting above

          float bias = w_.C_(0,0);
          blaze::forEach(A_, [=](float x) { return x + bias; });

          mblas::SafeSoftmax(A_);
          AlignedSourceContext = A_ * SourceContext;
        }

        void GetAttention(mblas::Matrix& Attention) {
          Attention = A_;
        }

        mblas::Matrix& GetAttention() {
          return A_;
        }

      private:
        const Weights& w_;

        mblas::Matrix SCU_;
        mblas::Matrix Temp1_;
        mblas::Matrix Temp2_;
        mblas::Matrix A_;
        mblas::ColumnVector V_;
    };

    //////////////////////////////////////////////////////////////
    template <class Weights>
    class Softmax {
      public:
        Softmax(const Weights& model)
        : w_(model),
        filtered_(false)
        {}

        void GetProbs(mblas::ArrayMatrix& Probs,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embedding,
                  const mblas::Matrix& AlignedSourceContext) {
          using namespace mblas;

          T1_ = State * w_.W1_;
          T2_ = Embedding * w_.W2_;
          T3_ = AlignedSourceContext * w_.W3_;

          AddBiasVector<byRow>(T1_, w_.B1_);
          AddBiasVector<byRow>(T2_, w_.B2_);
          AddBiasVector<byRow>(T3_, w_.B3_);

          auto t = blaze::forEach(T1_ + T2_ + T3_, Tanh());

          if(!filtered_) {
            Probs_ = t * w_.W4_;
            AddBiasVector<byRow>(Probs_, w_.B4_);
          } else {
            Probs_ = t * FilteredW4_;
            AddBiasVector<byRow>(Probs_, FilteredB4_);
          }
          mblas::Softmax(Probs_);
          Probs = blaze::forEach(Probs_, Log());
        }

        void Filter(const std::vector<size_t>& ids) {
          filtered_ = true;
          using namespace mblas;
          FilteredW4_ = Assemble<byColumn, Matrix>(w_.W4_, ids);
          FilteredB4_ = Assemble<byColumn, Matrix>(w_.B4_, ids);
        }

      private:
        const Weights& w_;
        bool filtered_;

        mblas::Matrix FilteredW4_;
        mblas::Matrix FilteredB4_;

        mblas::Matrix T1_;
        mblas::Matrix T2_;
        mblas::Matrix T3_;
        mblas::Matrix Probs_;
    };

  public:
    Decoder(const Weights& model)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, model.decGru1_),
      rnn2_(model.decGru2_),
	  attention_(model.decAttention_),
      softmax_(model.decSoftmax_)
    {}

    void Decode(mblas::Matrix& NextState,
                  const mblas::Matrix& State,
                  const mblas::Matrix& Embeddings,
                  const mblas::Matrix& SourceContext) {
      GetHiddenState(HiddenState_, State, Embeddings);
      GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext);
      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      GetProbs(NextState, Embeddings, AlignedSourceContext_);
    }

    BaseMatrix& GetProbs() {
      return Probs_;
    }

    void EmptyState(mblas::Matrix& State,
                    const mblas::Matrix& SourceContext,
                    size_t batchSize = 1) {
    	rnn1_.InitializeState(State, SourceContext, batchSize);
    	attention_.Init(SourceContext);
    }

    void EmptyEmbedding(mblas::Matrix& Embedding,
                        size_t batchSize = 1) {
      Embedding.resize(batchSize, embeddings_.GetCols());
      Embedding = 0.0f;
    }

    void Lookup(mblas::Matrix& Embedding,
                const std::vector<size_t>& w) {
      embeddings_.Lookup(Embedding, w);
    }

    void Filter(const std::vector<size_t>& ids) {
      softmax_.Filter(ids);
    }

    void GetAttention(mblas::Matrix& attention) {
    	attention_.GetAttention(attention);
    }

    mblas::Matrix& GetAttention() {
      return attention_.GetAttention();
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

    void GetAlignedSourceContext(mblas::Matrix& AlignedSourceContext,
                                 const mblas::Matrix& HiddenState,
                                 const mblas::Matrix& SourceContext) {
    	attention_.GetAlignedSourceContext(AlignedSourceContext, HiddenState, SourceContext);
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
    mblas::ArrayMatrix Probs_;

    Embeddings<Weights::Embeddings> embeddings_;
    RNNHidden<Weights::DecInit, Weights::GRU> rnn1_;
    RNNFinal<Weights::DecGRU2> rnn2_;
    Attention<Weights::DecAttention> attention_;
    Softmax<Weights::DecSoftmax> softmax_;
};

}
}

