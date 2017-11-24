#pragma once

#include "../mblas/matrix.h"
#include "model.h"
#include "gru.h"
#include "transition.h"
#include "common/god.h"

namespace amunmt {
namespace CPU {
namespace Nematus {

class Decoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
          : w_(model)
        {}

        void Lookup(mblas::Matrix& Rows, const std::vector<uint>& ids) {
          using namespace mblas;
          std::vector<uint> tids = ids;
          for (auto&& id : tids) {
            if (id >= w_.E_.rows()) {
              id = 1;
            }
          }
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
          : w_(initModel),
            gru_(gruModel)
        {}

        void InitializeState(
          mblas::Matrix& State,
          const mblas::Matrix& SourceContext,
          const size_t batchSize = 1)
        {
          using namespace mblas;

          // Calculate mean of source context, rowwise
          // Repeat mean batchSize times by broadcasting
          Temp1_ = Mean<byRow, Matrix>(SourceContext);

          Temp2_.resize(batchSize, SourceContext.columns());
          Temp2_ = 0.0f;
          AddBiasVector<byRow>(Temp2_, Temp1_);

          State = Temp2_ * w_.Wi_;
          AddBiasVector<byRow>(State, w_.Bi_);

          if (w_.lns_.rows()) {
            LayerNormalization(State, w_.lns_, w_.lnb_);
          }
          State = blaze::forEach(State, Tanh());
          // std::cerr << "INIT: " << std::endl;
          // for (int i = 0; i < 5; ++i) std::cerr << State(0, i) << " ";
          // std::cerr << std::endl;
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
    template <class WeightsGRU, class WeightsTrans>
    class RNNFinal {
      public:
        RNNFinal(const WeightsGRU& modelGRU, const WeightsTrans& modelTrans)
          : gru_(modelGRU),
            transition_(modelTrans)
        {}

        void GetNextState(
          mblas::Matrix& nextState,
          const mblas::Matrix& state,
          const mblas::Matrix& context)
        {
          gru_.GetNextState(nextState, state, context);
          transition_.GetNextState(nextState);
          // std::cerr << "TRANS: " << std::endl;
          // for (int i = 0; i < 10; ++i) std::cerr << nextState(0, i) << " ";
          // std::cerr << std::endl;
        }

      private:
        const GRU<WeightsGRU> gru_;
        const Transition transition_;
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
          mblas::AddBiasVector<mblas::byRow>(SCU_, w_.B_);

          if (w_.Wc_att_lns_.rows()) {
            LayerNormalization(SCU_, w_.Wc_att_lns_, w_.Wc_att_lnb_);
          }
        }

        void GetAlignedSourceContext(
          mblas::Matrix& AlignedSourceContext,
          const mblas::Matrix& HiddenState,
          const mblas::Matrix& SourceContext)
        {
          using namespace mblas;

          Temp2_ = HiddenState * w_.W_;
          if (w_.W_comb_lns_.rows()) {
            LayerNormalization(Temp2_, w_.W_comb_lns_, w_.W_comb_lnb_);
          }

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
          AddBiasVector<byRow>(T1_, w_.B1_);
          if (w_.lns_1_.rows()) {
            LayerNormalization(T1_, w_.lns_1_, w_.lnb_1_);
          }
          // std::cerr << "State" << std::endl;
          // for(int i = 0; i < 5; ++i) std::cerr << T1_(0, i) << " ";
          // std::cerr << std::endl;

          T2_ = Embedding * w_.W2_;
          AddBiasVector<byRow>(T2_, w_.B2_);
          if (w_.lns_2_.rows()) {
            LayerNormalization(T2_, w_.lns_2_, w_.lnb_2_);
          }
          // std::cerr << "emb" << std::endl;
          // for(int i = 0; i < 5; ++i) std::cerr << T2_(0, i) << " ";
          // std::cerr << std::endl;

          T3_ = AlignedSourceContext * w_.W3_;
          AddBiasVector<byRow>(T3_, w_.B3_);
          if (w_.lns_3_.rows()) {
            LayerNormalization(T3_, w_.lns_3_, w_.lnb_3_);
          }
          // std::cerr << "CTX" << std::endl;
          // for(int i = 0; i < 5; ++i) std::cerr << T3_(0, i) << " ";
          // std::cerr << std::endl;

          auto t = blaze::forEach(T1_ + T2_ + T3_, Tanh());

          if(!filtered_) {
            Probs = t * w_.W4_;
            AddBiasVector<byRow>(Probs, w_.B4_);
          } else {
            Probs = t * FilteredW4_;
            AddBiasVector<byRow>(Probs, FilteredB4_);
          }
          // std::cerr << "LOgit" << std::endl;
          // for(int i = 0; i < 5; ++i) std::cerr << Probs(0, i) << " ";
          // std::cerr << std::endl;
          LogSoftmax(Probs);
        }

        void Filter(const std::vector<uint>& ids) {
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
    };

  public:
    Decoder(const Weights& model)
    : embeddings_(model.decEmbeddings_),
      rnn1_(model.decInit_, model.decGru1_),
      rnn2_(model.decGru2_, model.decTransition_),
      attention_(model.decAttention_),
      softmax_(model.decSoftmax_)
    {}

    void Decode(
      mblas::Matrix& NextState,
      const mblas::Matrix& State,
      const mblas::Matrix& Embeddings,
      const mblas::Matrix& SourceContext)
    {
      GetHiddenState(HiddenState_, State, Embeddings);
      // std::cerr << "HIDDEN: " << std::endl;
      // for (int i = 0; i < 5; ++i) std::cerr << HiddenState_(0, i) << " ";
      // std::cerr << std::endl;

      GetAlignedSourceContext(AlignedSourceContext_, HiddenState_, SourceContext);
      // std::cerr << "ALIGNED SRC: " << std::endl;
      // for (int i = 0; i < 5; ++i) std::cerr << AlignedSourceContext_(0, i) << " ";
      // std::cerr << std::endl;

      GetNextState(NextState, HiddenState_, AlignedSourceContext_);
      // std::cerr << "NEXT: " << std::endl;
      // for (int i = 0; i < 5; ++i) std::cerr << NextState(0, i) << " ";
      // std::cerr << std::endl;

      GetProbs(NextState, Embeddings, AlignedSourceContext_);
    }

    mblas::ArrayMatrix& GetProbs() {
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
                const std::vector<uint>& w) {
      embeddings_.Lookup(Embedding, w);
    }

    void Filter(const std::vector<uint>& ids) {
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
    RNNFinal<Weights::DecGRU2, Weights::Transition> rnn2_;
    Attention<Weights::DecAttention> attention_;
    Softmax<Weights::DecSoftmax> softmax_;
};

}
}
}

