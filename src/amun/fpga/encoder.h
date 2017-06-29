#pragma once

#include "model.h"
#include "matrix.h"
#include "gru.h"
#include "array.h"
#include "common/sentences.h"

namespace amunmt {
namespace FPGA {

class Encoder {
  template <class Weights>
  class Embeddings {
  public:
    Embeddings(const Weights& model)
    : w_(model)
    {}

    void Lookup(const OpenCLInfo &openCLInfo, mblas::Matrix& Row, const Words& words)
    {
      std::vector<uint> knownWords(words.size(), 1);
      for (size_t i = 0; i < words.size(); ++i) {
        if (words[i] < w_.E_.dim(0)) {
          knownWords[i] = words[i];
        }
      }

      Array<uint> dKnownWords(openCLInfo, knownWords);

      /*
      std::cerr << "dKnownWords=" << dKnownWords.Debug(1) << " std::vector=" << mblas::Sum(knownWords) << ": ";
      for (size_t i = 0; i < knownWords.size(); ++i) {
        std::cerr << knownWords[i] << " ";
      }
      std::cerr << std::endl;
      */

      //std::cerr << "Row1=" << Row.Debug(1) << std::endl;
      Row.Resize(words.size(), w_.E_.dim(1));
      //std::cerr << "Row2=" << Row.Debug(1) << std::endl;
      mblas::Assemble(Row, w_.E_, dKnownWords);

      //std::cerr << "Row3=" << Row.Debug(1) << std::endl;

    }

  private:
    const Weights& w_;
  };

  template <class Weights>
  class RNN {
    public:
    public:
      RNN(const OpenCLInfo &openCLInfo, const Weights& model)
      : openCLInfo_(openCLInfo)
      , gru_(openCLInfo, model)
      , State_(openCLInfo)
    {}

    size_t GetStateLength() const {
      return gru_.GetStateLength();
    }

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
    void Encode(It it, It end, mblas::Matrix& Context, size_t batchSize, bool invert,
                    const Array<int>* mapping=nullptr)
    {
      InitializeState(batchSize);

      mblas::Matrix prevState(State_);
      //std::cerr << "State_=" << State_.Debug(1) << std::endl;
      //std::cerr << "prevState=" << prevState.Debug(1) << std::endl;

      size_t n = std::distance(it, end);
      size_t i = 0;

      while(it != end) {
        GetNextState(State_, prevState, *it++);

        //std::cerr << "invert=" << invert << std::endl;
        if(invert) {
          assert(mapping);

          //std::cerr << "1State_=" << State_.Debug(1) << std::endl;
          //std::cerr << "mapping=" << mapping->Debug(1) << std::endl;
          mblas::MapMatrix(State_, *mapping, n - i - 1);
          //std::cerr << "2State_=" << State_.Debug(1) << std::endl;

          mblas::PasteRows(Context, State_, (n - i - 1), gru_.GetStateLength(), n);
        }
        else {
          //std::cerr << "1Context=" << Context.Debug(1) << std::endl;
          mblas::PasteRows(Context, State_, i, 0, n);
          //std::cerr << "2Context=" << Context.Debug(1) << std::endl;
        }

        prevState.Swap(State_);
        ++i;
      }

    }

    private:
      const OpenCLInfo &openCLInfo_;

      // Model matrices
      const GRU<Weights> gru_;
      mblas::Matrix State_;

  };

public:
  Encoder(const OpenCLInfo &openCLInfo, const Weights& model);

  void Encode(const Sentences& source, size_t tab, mblas::Matrix& Context,
                Array<int>& dMapping);

protected:
  Embeddings<Weights::EncEmbeddings> embeddings_;
  RNN<Weights::EncForwardGRU> forwardRnn_;
  RNN<Weights::EncBackwardGRU> backwardRnn_;

  // reusing memory
  std::vector<mblas::Matrix> embeddedWords_;
  mblas::Matrix Context;

  const OpenCLInfo &openCLInfo_;

};

}
}
