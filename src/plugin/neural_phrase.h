#pragma once

#include <vector>
#include <string>

class NeuralPhrase {
  public:
    NeuralPhrase() {}

    NeuralPhrase(const std::vector<std::string>& words, std::vector<float> scores,
                 int startPos, int endPos)
      : words(words), scores(scores), startPos(startPos), endPos(endPos) {}

    std::vector<std::string>& getWords() {
      return words;
    }

    float getScore(int i) {
      return scores[i];
    }

    std::pair<int, int> getCoverage() {
      return std::make_pair(startPos, endPos);
    }

  private:
   std::vector<std::string> words;
   std::vector<float> scores;
   int startPos, endPos;

};
