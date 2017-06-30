#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "3rd_party/exception.h"
#include "common/definitions.h"
#include "common/utils.h"
#include "data/vocab.h"
#include "data/types.h"
#include "layers/param_initializers.h"

namespace marian {

class EmbeddingReader {
public:
  EmbeddingReader() {}

  std::vector<float> read(const std::string& fileName,
                          int dimVoc,
                          int dimEmb) {
    std::ifstream embFile(fileName);
    UTIL_THROW_IF2(!embFile.is_open(),
                   "Unable to open file with embeddings: " + fileName);
    std::string line;

    // Read the first line with two values: a number of words in the vocabulary
    // and the length of embedding vector
    std::getline(embFile, line);
    std::vector<std::string> values;
    Split(line, values);
    UTIL_THROW_IF2(values.size() != 2,
                   "Unexpected format of the first line in file with embeddings");
    UTIL_THROW_IF2(stoi(values[1]) != dimEmb,
                   "Unexpected length of embedding vectors");

    // Read embedding vectors of words present in the vocabulary
    std::unordered_map<Word, std::vector<float>> word2vec;
    while(std::getline(embFile, line)) {
      values.clear();
      Split(line, values);

      Word word = std::stoi(values.front());

      // Skip unknown words, but not <unk> word
      if(word == UNK_ID && values.front() != UNK_STR)
        continue;

      std::transform(values.begin() + 1,
                     values.end(),
                     word2vec[word].begin(),
                     [](const std::string& s) { return std::stof(s); });
    }

    std::vector<float> embs;
    embs.reserve(dimVoc * dimEmb);

    // Populate output vector with embedding
    for(size_t word = 0; word < (size_t)dimVoc; ++word) {
      auto pos = embs.begin() + (word * dimEmb);

      if(word2vec.find(word) != word2vec.end()) {
        embs.insert(pos, word2vec[word].begin(), word2vec[word].end());
      } else {
        // For words not occuring in the file use uniform distribution
        std::vector<float> values;
        values.reserve(dimEmb);
        // @TODO: consider generating values once for all missing words and
        // then use the generated numbers to bucket into embedding vectors
        inits::distribution<std::uniform_real_distribution<float>>(
            values, -0.1, 0.1);
        embs.insert(pos, values.begin(), values.end());
      }
    }

    return embs;
  }
};

}  // namespace marian
