#pragma once

#include "model.h"
#include "matrix.h"
#include "matrix_functions.h"
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

      Array<uint> dKnownWords(openCLInfo.context, openCLInfo.device, knownWords);

      /*
      std::cerr << "dKnownWords=" << dKnownWords.Debug(true) << " std::vector=" << mblas::Sum(knownWords) << ": ";
      for (size_t i = 0; i < knownWords.size(); ++i) {
        std::cerr << knownWords[i] << " ";
      }
      std::cerr << std::endl;
      */

      //std::cerr << "Row1=" << Row.Debug(true) << std::endl;
      Row.Resize(words.size(), w_.E_.dim(1));
      //std::cerr << "Row2=" << Row.Debug(true) << std::endl;
      mblas::Assemble(openCLInfo, Row, w_.E_, dKnownWords);

      std::cerr << "Row3=" << Row.Debug(true) << std::endl;

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
      mblas::Fill(openCLInfo_, State_, 0.0f);
    }

    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Embd) {
      gru_.GetNextState(NextState, State, Embd);
    }

    template <class It>
    void GetContext(It it, It end, mblas::Matrix& Context, size_t batchSize, bool invert)
    {
      InitializeState(batchSize);

      mblas::Matrix prevState(State_);
      std::cerr << "State_=" << State_.Debug(true) << std::endl;
      std::cerr << "prevState=" << prevState.Debug(true) << std::endl;

      size_t n = std::distance(it, end);
      size_t i = 0;

      while(it != end) {
        GetNextState(State_, prevState, *it++);

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

  void GetContext(const Sentences& source, size_t tab, mblas::Matrix& Context);

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
