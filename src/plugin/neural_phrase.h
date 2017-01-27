#pragma once

#include <vector>
#include <string>

namespace amunmt {

class NeuralPhrase {
  public:
   std::vector<size_t> words;
   std::vector<float> scores;
   int startPos, endPos;

    NeuralPhrase() {}

    NeuralPhrase(const std::vector<size_t>& words, std::vector<float> scores,
                 int startPos, int endPos)
      : words(words), scores(scores), startPos(startPos), endPos(endPos) {}

    float getScore(int i) {
      return scores[i];
    }

    std::pair<int, int> getCoverage() {
      return std::make_pair(startPos, endPos);
    }

    std::string Debug() const;

  private:

};

}

