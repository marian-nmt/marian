#pragma once

#include "gpu/mblas/matrix_functions.h"
#include "model.h"
#include "gru.h"
#include "common/sentence.h"

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
          // mblas::Debug(Row);
        }

      private:
        const Weights& w_;
    };

    template <class Weights>
    class RNN {
      public:
        RNN(const Weights& model)
        : gru_(model) {}

        void InitializeState(size_t batchSize = 1) {
          // std::cerr << "BATCH: " << batchSize  << "; " << gru_.GetStateLength() << std::endl;
          State_.Resize(batchSize, gru_.GetStateLength());
          mblas::Fill(State_, 0.0f);
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }

        template <class It>
        void GetContext(It it, It end, mblas::Matrix& Context, size_t batchSize, bool invert) {
          // std::cerr << "INIT" << std::endl;
          InitializeState(batchSize);

          size_t n = std::distance(it, end);
          // std::cerr << "N: " << n << std::endl;
          size_t i = 0;
          while(it != end) {
            // std::cerr << "generating: " << i  << std::endl;
            GetNextState(State_, State_, *it++);
            // std::cerr << "pasting " << std::endl;
            // std::cerr << "STATE: " << State_.Rows() << " x " << State_.Cols() << std::endl;
            // std::cerr << "CONTEXT: " << Context.Rows() << " x " << Context.Cols() << std::endl;
            // std::cerr << Context.GetVec().back() << std::endl;
            // mblas::Debug(Context);
            // std::cerr << "DEBUG DONE";
            // mblas::Debug(State_);
            // std::cerr << "DEBUG DONE";
            if(invert)
              mblas::PasteRows(Context, State_, (n - i - 1) * batchSize, gru_.GetStateLength(), n);
            else
              mblas::PasteRows(Context, State_, i * batchSize, 0, n);
            // std::cerr << "next" << std::endl;
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

    void GetContext(const Sentences& words, size_t tab, mblas::Matrix& Context);

  private:
    Embeddings<Weights::EncEmbeddings> embeddings_;
    RNN<Weights::EncForwardGRU> forwardRnn_;
    RNN<Weights::EncBackwardGRU> backwardRnn_;

    // reusing memory
    std::vector<mblas::Matrix> embeddedWords_;
};

}

