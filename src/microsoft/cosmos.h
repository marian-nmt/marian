#pragma once

#include <memory>
#include <string>
#include <vector>

namespace marian {

template <typename T>
using Ptr = std::shared_ptr<T>;

namespace cosmos {
  class Embedder;

  /**
   * MarianEmbedder takes a Marian sequence2sequence transformer model and produces
   * sentence embeddings collected from the encoder. Currently the model file is supposed 
   * to know how to do that. 
   */
  class MarianEmbedder {
    private:
      Ptr<Embedder> embedder_;

    public:
      MarianEmbedder();

      /**
       * `input` is a big string with multiple sentences separated by '\n'.
       * Returns a vector of embedding vectors in order corresponding to input sentence order.
       */
      std::vector<std::vector<float>> embed(const std::string& input);

      /** 
       * `modelPath` is a Marian model, `vocabPath` a matching SentencePiece model with *.spm suffix.
       */
      bool load(const std::string& modelPath, const std::string& vocabPath);
  };

  /**
   * MarianCosineScorer takes a Marian sequence2sequence transformer model and produces
   * sentence-wise cosine similarities for two sentence embeddings.
   */
  class MarianCosineScorer {
    private:
      Ptr<Embedder> embedder_;

    public:
      MarianCosineScorer();

      /**
       * `input1` and `input2' are big strings with multiple sentences separated by '\n'.
       * Both inputs have to have the same number of separated lines.
       * Returns a vector of similarity scores in order corresponding to input sentence order.
       */
      std::vector<float> score(const std::string& input1, const std::string& input2);
      
      /** 
       * `modelPath` is a Marian model, `vocabPath` a matching SentencePiece model with *.spm suffix.
       */
      bool load(const std::string& modelPath, const std::string& vocabPath);
  };
}

}