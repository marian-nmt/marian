#pragma once

#include <yaml-cpp/yaml.h>

#include "gpu/mblas/matrix_functions.h"
#include "model.h"
#include "gru.h"
#include "common/sentence.h"
#include "gpu/types-gpu.h"
#include "gru.h"
#include "lstm.h"
#include "multiplicative.h"
#include "cell.h"
#include "cellstate.h"

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
          std::vector<uint> knownWords(words.size(), 1);
          for (size_t i = 0; i < words.size(); ++i) {
            if (words[i] < w_.E_->dim(0)) {
              knownWords[i] = words[i];
            }
          }

          mblas::Array<uint> dKnownWords(knownWords);

          Row.NewSize(words.size(), w_.E_->dim(1));
          mblas::Assemble(Row, *w_.E_, dKnownWords);
          //std::cerr << "Row3=" << Row.Debug(1) << std::endl;
        }

      private:
        const Weights& w_;

        Embeddings(const Embeddings&) = delete;
    };

    class RNN {
      public:
        RNN(std::unique_ptr<Cell> cell)
          : gru_(std::move(cell)) {}

        void InitializeState(size_t batchSize = 1) {
          CellLength cellLength = gru_->GetStateLength();
          if (cellLength.cell > 0) {
            State_.cell->NewSize(batchSize, cellLength.cell);
            mblas::Fill(*(State_.cell), 0.0f);
          }
          State_.output->NewSize(batchSize, cellLength.output);
          mblas::Fill(*(State_.output), 0.0f);
        }

        void GetNextState(CellState& NextState,
                          const CellState& State,
                          const mblas::Matrix& Embd) {
          gru_->GetNextState(NextState, State, Embd);
        }

        template <class It>
        void Encode(It it, It end, mblas::Matrix& Context,
                    size_t batchSize, bool invert,
                    const mblas::IMatrix *sentenceLengths=nullptr)
        {
          InitializeState(batchSize);

          CellState prevState(std::unique_ptr<mblas::Matrix>(new mblas::Matrix(*(State_.cell))),
                              std::unique_ptr<mblas::Matrix>(new mblas::Matrix(*(State_.output))));
          size_t n = std::distance(it, end);
          size_t i = 0;

          while(it != end) {
            GetNextState(State_, prevState, *it++);

            //std::cerr << "invert=" << invert << std::endl;
            if(invert) {
              assert(sentenceLengths);

              //std::cerr << "1State_=" << State_.Debug(1) << std::endl;
              //std::cerr << "mapping=" << mblas::Debug(*mapping) << std::endl;
              //mblas::MapMatrix(*(State_.cell), *sentencesMask, n - i - 1);
              mblas::MapMatrix(*(State_.output), *sentenceLengths, n - i - 1);
              if (State_.cell->size()) {
                mblas::MapMatrix(*(State_.cell), *sentenceLengths, n - i - 1);
              }
              //std::cerr << "2State_=" << State_.Debug(1) << std::endl;

              mblas::PasteRows(Context, *(State_.output), (n - i - 1), gru_->GetStateLength().output);
            }
            else {
              //std::cerr << "1Context=" << Context.Debug(1) << std::endl;
              mblas::PasteRows(Context, *(State_.output), i, 0);
              //std::cerr << "2Context=" << Context.Debug(1) << std::endl;
            }

            if (State_.cell->size() > 0) {
              prevState.cell->swap(*(State_.cell));
            }
            prevState.output->swap(*(State_.output));
            ++i;
          }
        }

        CellLength GetStateLength() const {
          return gru_->GetStateLength();
        }

      private:
        const std::unique_ptr<Cell> gru_;
        CellState State_;
        RNN(const RNN&) = delete;
    };

  public:
    Encoder(const Weights& model, const YAML::Node& config);

    void Encode(const Sentences& words, size_t tab, mblas::Matrix& context,
                    mblas::IMatrix &sentenceLengths);

  private:
    std::unique_ptr<Cell> InitForwardCell(const Weights& model, const YAML::Node& config);
    std::unique_ptr<Cell> InitBackwardCell(const Weights& model, const YAML::Node& config);

  private:
    Embeddings<Weights::EncEmbeddings> embeddings_;
    RNN forwardRnn_;
    RNN backwardRnn_;

    // reusing memory
    std::vector<mblas::Matrix> embeddedWords_;

    Encoder(const Encoder&) = delete;
};

}
}

