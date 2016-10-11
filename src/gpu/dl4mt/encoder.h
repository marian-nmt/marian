#pragma once

#include "gpu/mblas/matrix.h"
#include "model.h"
#include "gru.h"

namespace GPU {

class Encoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Row, size_t i) {
          using namespace mblas;
          if(i < w_.E_.Rows())
            CopyRow(Row, w_.E_, i);
          else
            CopyRow(Row, w_.E_, 1); // UNK
        }

        const Weights& w_;
      private:
    };

    template <class Weights>
    class RNN {
      public:
        RNN(const Weights& model)
        : gru_(model) {}

        void InitializeState(size_t batchSize = 1) {
          State_.Clear();
          State_.Resize(batchSize, gru_.GetStateLength(), 0.0);
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }

        template <class It>
        void GetContext(It it, It end, 
                        mblas::Matrix& Context, bool invert) {
          InitializeState();

          size_t n = std::distance(it, end);
          size_t i = 0;
          while(it != end) {
            GetNextState(State_, State_, *it++);
            if(invert)
              mblas::PasteRow(Context, State_, n - i - 1, gru_.GetStateLength());
            else
              mblas::PasteRow(Context, State_, i, 0);
            ++i;
          }
        }

        size_t GetStateLength() const {
          return gru_.GetStateLength();
        }

      private:
        // Model matrices
        const GRU<Weights> gru_;

        mblas::Matrix State_;
    };

  public:
    Encoder(const Weights& model);

    void GetContext(const std::vector<size_t>& words,
                    mblas::Matrix& Context);

  private:
    Embeddings<Weights::EncEmbeddings> embeddings_;
    RNN<Weights::EncForwardGRU> forwardRnn_;
    RNN<Weights::EncBackwardGRU> backwardRnn_;
};

}

