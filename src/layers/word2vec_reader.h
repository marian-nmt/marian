#pragma once

#include "marian.h"

#include "common/logging.h"

#include <fstream>
#include <string>
#include <vector>

namespace marian {

class Word2VecReader {
public:
  Word2VecReader() {}

  std::vector<float> read(const std::string& fileName, int dimVoc, int dimEmb) {
    LOG(info, "[data] Loading embedding vectors from {}", fileName);

    io::InputFileStream embFile(fileName);

    std::string line;
    std::vector<std::string> values;
    values.reserve(dimEmb);

    // The first line contains two values: the number of words in the
    // vocabulary and the length of embedding vectors
    io::getline(embFile, line);
    utils::split(line, values);
    ABORT_IF(values.size() != 2,
             "Unexpected format of the first line of the embedding file");
    ABORT_IF(stoi(values[1]) != dimEmb,
             "Unexpected length of embedding vectors");

    // Read embedding vectors into a map
    std::unordered_map<WordIndex, std::vector<float>> word2vec;
    while(io::getline(embFile, line)) {
      values.clear();
      utils::split(line, values);

      WordIndex word = std::stoi(values.front());
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
    for(WordIndex word = 0; word < (WordIndex)dimVoc; ++word) {
      // For words not occuring in the file use uniform distribution
      if(word2vec.find(word) == word2vec.end()) {
        auto randVals = randomEmbeddings(dimVoc, dimEmb);
        embs.insert(embs.end(), randVals.begin(), randVals.end());
      } else {
        embs.insert(embs.end(), word2vec[word].begin(), word2vec[word].end());
      }
    }

    embs.resize(dimVoc * dimEmb, 0); // @TODO: is it correct to zero out the remaining embeddings?
    return embs;
  }

private:
  std::vector<float> randomEmbeddings(int dimVoc, int dimEmb) {
    std::vector<float> values;
    values.resize(dimEmb);
    // Glorot numal distribution
    float scale = sqrtf(2.0f / (dimVoc + dimEmb));

    // @TODO: switch to new random generator back-end.
    // This is rarely used however.
    std::random_device rd;
    std::mt19937 engine(rd());

    std::normal_distribution<float> d(0, scale);
    auto gen = [&d, &engine] () {
       return d(engine);
    };

    std::generate(values.begin(), values.end(), gen);

    return values;
  }
};

}  // namespace marian
