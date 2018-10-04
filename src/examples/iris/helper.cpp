#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/config.h"
#include "common/utils.h"

using namespace marian;

// Constants for Iris example
const int NUM_FEATURES = 4;
const int NUM_LABELS = 3;

void readIrisData(const std::string fileName,
                  std::vector<float>& features,
                  std::vector<IndexType>& labels) {
  std::map<std::string, IndexType> CLASSES
      = {{"Iris-setosa", 0}, {"Iris-versicolor", 1}, {"Iris-virginica", 2}};

  std::ifstream in(fileName);
  if(!in.is_open()) {
    std::cerr << "Iris dataset not found: " << fileName << std::endl;
  }
  std::string line;
  std::string value;
  while(std::getline(in, line)) {
    std::stringstream ss(line);
    int i = 0;
    while(std::getline(ss, value, ',')) {
      if(++i == 5)
        labels.emplace_back(CLASSES[value]);
      else
        features.emplace_back(std::stof(value));
    }
  }
}

void shuffleData(std::vector<float>& features, std::vector<IndexType>& labels) {
  // Create a list of indices 0...K
  std::vector<size_t> indices;
  indices.reserve(labels.size());
  for(size_t i = 0; i < labels.size(); ++i)
    indices.push_back(i);

  // Shuffle indices
  static std::mt19937 urng(marian::Config::seed);
  std::shuffle(indices.begin(), indices.end(), urng);

  std::vector<float> featuresTemp;
  featuresTemp.reserve(features.size());
  std::vector<IndexType> labelsTemp;
  labelsTemp.reserve(labels.size());

  // Get shuffled features and labels
  for(size_t i = 0; i < indices.size(); ++i) {
    auto idx = indices[i];
    labelsTemp.push_back(labels[idx]);
    featuresTemp.insert(featuresTemp.end(),
                        features.begin() + (idx * NUM_FEATURES),
                        features.begin() + ((idx + 1) * NUM_FEATURES));
  }

  features = featuresTemp;
  labels = labelsTemp;
}

float calculateAccuracy(const std::vector<float> probs,
                        const std::vector<IndexType> labels) {
  size_t numCorrect = 0;
  for(size_t i = 0; i < probs.size(); i += NUM_LABELS) {
    auto pred = std::distance(
        probs.begin() + i,
        std::max_element(probs.begin() + i, probs.begin() + i + NUM_LABELS));
    if(pred == labels[i / NUM_LABELS])
      ++numCorrect;
  }
  return numCorrect / float(labels.size());
}
