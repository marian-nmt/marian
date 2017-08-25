#pragma once

#include "gpu/mblas/matrix_functions.h"
#include "model.h"
#include "gru.h"
#include "common/sentence.h"
#include "gpu/types-gpu.h"

namespace amunmt {

class Sentences;

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
          HostVector<uint> knownWords(words.size(), 1);
          for (size_t i = 0; i < words.size(); ++i) {
            if (words[i] < w_.E_->dim(0)) {
              knownWords[i] = words[i];
            }
          }

          DeviceVector<uint> dKnownWords(knownWords);

          Row.NewSize(words.size(), w_.E_->dim(1));
          mblas::Assemble(Row, *w_.E_, dKnownWords);
          //std::cerr << "Row3=" << Row.Debug(1) << std::endl;
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
          State_.NewSize(batchSize, gru_.GetStateLength());
          mblas::Fill(State_, 0.0f);
        }

        void GetNextState(mblas::Matrix& NextState,
                          const mblas::Matrix& State,
                          const mblas::Matrix& Embd) {
          gru_.GetNextState(NextState, State, Embd);
        }

        template <class It>
        void Encode(It it, It end, mblas::Matrix& Context, size_t batchSize, bool invert,
                        const mblas::IMatrix *sentencesMask=nullptr)
        {
          InitializeState(batchSize);

          mblas::Matrix prevState(State_);
          size_t n = std::distance(it, end);
          size_t i = 0;

          while(it != end) {
            GetNextState(State_, prevState, *it++);
	    
            //std::cerr << "invert=" << invert << std::endl;
            if(invert) {
              assert(sentencesMask);

              //std::cerr << "1State_=" << State_.Debug(1) << std::endl;
              //std::cerr << "mapping=" << mblas::Debug(*mapping) << std::endl;
              mblas::MapMatrix(State_, *sentencesMask, n - i - 1);
              //std::cerr << "2State_=" << State_.Debug(1) << std::endl;

              mblas::PasteRows(Context, State_, (n - i - 1), gru_.GetStateLength());
            }
            else {
              //std::cerr << "1Context=" << Context.Debug(1) << std::endl;
              mblas::PasteRows(Context, State_, i, 0);
              //std::cerr << "2Context=" << Context.Debug(1) << std::endl;
            }

            prevState.swap(State_);
            ++i;
          }
        }

        size_t GetStateLength() const {
          return gru_.GetStateLength();
        }

      private:
        const GRU<Weights> gru_;
        mblas::Matrix State_;
        RNN(const RNN&) = delete;
    };

  public:
    Encoder(const Weights& model);

    void Encode(const Sentences& words, size_t tab, mblas::Matrix& context,
                    mblas::IMatrix &sentencesMask);

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

