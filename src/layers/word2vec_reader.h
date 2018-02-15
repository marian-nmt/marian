#pragma once

#include "3rd_party/exception.h"
#include "marian.h"

#include <fstream>
#include <string>
#include <vector>

namespace marian {

class Word2VecReader {
public:
  Word2VecReader() {}

  std::vector<float> read(const std::string& fileName, int dimVoc, int dimEmb) {
    LOG(info, "[data] Loading embedding vectors from {}", fileName);

    std::ifstream embFile(fileName);
    ABORT_IF(!embFile.is_open(),
             "Unable to open file with embeddings: " + fileName);

    std::string line;
    std::vector<std::string> values;
    values.reserve(dimEmb);

    // The first line contains two values: the number of words in the
    // vocabulary and the length of embedding vectors
    std::getline(embFile, line);
    Split(line, values);
    ABORT_IF(values.size() != 2,
             "Unexpected format of the first line in embedding file");
    ABORT_IF(stoi(values[1]) != dimEmb,
             "Unexpected length of embedding vectors");

    // Read embedding vectors into a map
    std::unordered_map<Word, std::vector<float>> word2vec;
    while(std::getline(embFile, line)) {
      values.clear();
      Split(line, values);

      Word word = std::stoi(values.front());
      if(word >= (size_t)dimVoc)
        continue;

      word2vec[word].reserve(dimEmb);
      std::transform(values.begin() + 1,
                     values.end(),
                     std::back_inserter(word2vec[word]),
                     [](const std::string& s) { return std::stof(s); });
    }

    // Initialize final flat vector for embeddings
    std::vector<float> embs;
    embs.reserve(dimVoc * dimEmb);

    // Populate output vector with embedding
    for(size_t word = 0; word < (size_t)dimVoc; ++word) {
      // For words not occuring in the file use uniform distribution
      if(word2vec.find(word) == word2vec.end()) {
        auto randVals = randomEmbeddings(dimVoc, dimEmb);
        embs.insert(embs.end(), randVals.begin(), randVals.end());
      } else {
        embs.insert(embs.end(), word2vec[word].begin(), word2vec[word].end());
      }
    }

    return embs;
  }

private:
  std::vector<float> randomEmbeddings(int dimVoc, int dimEmb) {
    std::vector<float> values;
    values.reserve(dimEmb);
    // Glorot numal distribution
    float scale = sqrtf(2.0f / (dimVoc + dimEmb));
    inits::distribution<std::normal_distribution<float>>(values, 0, scale);
    return values;
  }
};

}  // namespace marian
