#pragma once

#include "../mblas/matrix.h"
#include "model.h"
#include "gru.h"
#include "transition.h"

namespace amunmt {
namespace CPU {
namespace Nematus {

class Encoder {
  private:

	/////////////////////////////////////////////////////////////////
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Row, size_t i) {
		  size_t len = w_.E_.columns();
          if(i < w_.E_.rows())
            Row = blaze::submatrix(w_.E_, i, 0, 1, len);
          else
            Row = blaze::submatrix(w_.E_, 1, 0, 1, len); // UNK
        }

        const Weights& w_;
      private:
    };

    /////////////////////////////////////////////////////////////////
    template <class WeightsGRU, class WeightsTrans>
    class EncoderRNN {
      public:
        EncoderRNN(const WeightsGRU& modelGRU, const WeightsTrans& modelTrans)
          : gru_(modelGRU),
            transition_(modelTrans)
        {}

        void InitializeState(size_t batchSize = 1) {
          State_.resize(batchSize, gru_.GetStateLength());
          State_ = 0.0f;
        }

        void GetNextState(mblas::Matrix& nextState,
                          const mblas::Matrix& state,
                          const mblas::Matrix& embd) {
          gru_.GetNextState(nextState, state, embd);
          // std::cerr << "GRU: " << std::endl;
          // for (int i = 0; i < 10; ++i) std::cerr << nextState(0, i) << " ";
          // std::cerr << std::endl;
          transition_.GetNextState(nextState);
          // std::cerr << "TRANS: " << std::endl;
          // for (int i = 0; i < 10; ++i) std::cerr << nextState(0, i) << " ";
          // std::cerr << std::endl;
        }

        template <class It>
        void GetContext(It it, It end, mblas::Matrix& Context, bool invert) {
          InitializeState();

          size_t n = std::distance(it, end);
          size_t i = 0;
          while(it != end) {
          GetNextState(State_, State_, *it++);

          size_t len = gru_.GetStateLength();
            if(invert)
              blaze::submatrix(Context, n - i - 1, len, 1, len) = State_;
            else
      			  blaze::submatrix(Context, i, 0, 1, len) = State_;
            ++i;
          }
        }

        size_t GetStateLength() const {
          return gru_.GetStateLength();
        }

      private:
        // Model matrices
        const GRU<WeightsGRU> gru_;
        const Transition transition_;

        mblas::Matrix State_;
    };

  /////////////////////////////////////////////////////////////////
  public:
    Encoder(const Weights& model)
      : embeddings_(model.encEmbeddings_),
        forwardRnn_(model.encForwardGRU_, model.encForwardTransition_),
        backwardRnn_(model.encBackwardGRU_, model.encBackwardTransition_)
    {}

    void GetContext(const std::vector<uint>& words,
                    mblas::Matrix& context);

  private:
    Embeddings<Weights::Embeddings> embeddings_;
    EncoderRNN<Weights::GRU, Weights::Transition> forwardRnn_;
    EncoderRNN<Weights::GRU, Weights::Transition> backwardRnn_;
};

}
}
}

