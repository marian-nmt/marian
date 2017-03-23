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

    void Lookup(const cl_context &context, mblas::Matrix& Row, const Words& words)
    {
      std::vector<size_t> knownWords(words.size(), 1);
      for (size_t i = 0; i < words.size(); ++i) {
        if (words[i] < w_.E_.dim(0)) {
          knownWords[i] = words[i];
        }
      }

      std::vector<size_t> dKnownWords(knownWords);
      Array<size_t> dKnownWords2(context, knownWords);

      Row.Resize(words.size(), w_.E_.dim(1));
      mblas::Assemble(Row, w_.E_, dKnownWords);

    }

  private:
    const Weights& w_;
  };

  template <class Weights>
  class RNN {
    public:
    public:
      RNN(const cl_context &context, const cl_device_id &device, const Weights& model)
      : gru_(context, device, model)
    {}

    size_t GetStateLength() const {
      return gru_.GetStateLength();
    }

    template <class It>
    void GetContext(It it, It end, mblas::Matrix& Context, size_t batchSize, bool invert)
    {

    }

    private:
      // Model matrices
      const GRU<Weights> gru_;

  };

public:
  Encoder(const cl_context &context, const cl_device_id &device, const Weights& model);

  void GetContext(const Sentences& source, size_t tab, mblas::Matrix& Context);

protected:
  Embeddings<Weights::EncEmbeddings> embeddings_;
  RNN<Weights::EncForwardGRU> forwardRnn_;
  RNN<Weights::EncBackwardGRU> backwardRnn_;

  // reusing memory
  std::vector<mblas::Matrix> embeddedWords_;
  mblas::Matrix Context;

  const cl_context &context_;
  const cl_device_id &device_;

};

}
}
