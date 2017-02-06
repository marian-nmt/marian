#pragma once

#include "gpu/mblas/matrix_functions.h"
#include "model.h"
#include "gru.h"
#include "common/sentence.h"
#include "gpu/types-gpu.h"

namespace amunmt {
namespace GPU {

class Encoder {
  private:
    template <class Weights>
    class Embeddings {
      public:
        Embeddings(const Weights& model)
        : w_(model)
        {}

        void Lookup(mblas::Matrix& Row, const Words& words) {
          thrust::host_vector<size_t> knownWords(words.size(), 1);
          for (size_t i = 0; i < words.size(); ++i) {
            if (words[i] < w_.E_.Rows()) {
              knownWords[i] = words[i];
            }
          }

          DeviceVector<size_t> dKnownWords(knownWords);

          Row.Resize(words.size(), w_.E_.Cols());
          mblas::Assemble(Row, w_.E_, dKnownWords);
        }

      private:
        const Weights& w_;

        Embeddings(const Embeddings&) = delete;
    };

    template <class Weights>
    class RNN {
      public:
        RNN(const Weights& model)
        : gru_(model) {}

        void InitializeState(size_t batchSize = 1) {
          State_.Resize(batchSize, gru_.GetStateLength());
          mblas::Fill(State_, 0.0f);
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }

        template <class It>
        void GetContext(It it, It end, mblas::Matrix& Context, size_t batchSize, bool invert,
                        const DeviceVector<int>* mapping=nullptr) {
          InitializeState(batchSize);

          size_t n = std::distance(it, end);
          size_t i = 0;
          while(it != end) {
            GetNextState(State_, State_, *it++);
            if(invert) {
              mblas::MapMatrix(State_, *mapping, n - i - 1);
              mblas::PasteRows(Context, State_, (n - i - 1), gru_.GetStateLength(), n);
            }
            else {
              mblas::PasteRows(Context, State_, i, 0, n);
            }
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

        RNN(const RNN&) = delete;
    };

  public:
    Encoder(const Weights& model);

    void GetContext(const Sentences& words, size_t tab, mblas::Matrix& Context,
                    DeviceVector<int>& mapping);

  private:
    Embeddings<Weights::EncEmbeddings> embeddings_;
    RNN<Weights::EncForwardGRU> forwardRnn_;
    RNN<Weights::EncBackwardGRU> backwardRnn_;

    // reusing memory
    std::vector<mblas::Matrix> embeddedWords_;

    Encoder(const Encoder&) = delete;
};

}
}

