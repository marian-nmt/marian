#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <sstream>
#include <functional>

#include <cuda.h>

#include "marian.h"

using namespace marian;
using namespace data;
using namespace keywords;


// Constants for Iris example
const size_t MAX_EPOCHS = 200;
const int NUM_FEATURES = 4;
const int NUM_LABELS = 3;


// Function creating feedforward dense network graph
Expr buildIrisClassifier(Ptr<ExpressionGraph> graph,
    std::vector<float> inputData,
    std::vector<float> outputData={},
    bool train=false) {

  // The number of input data
  int N = inputData.size() / NUM_FEATURES;

  graph->clear();

  // Define the input layer
  auto x = graph->constant({N, NUM_FEATURES}, init=inits::from_vector(inputData));

  // Define the hidden layer
  auto W1 = graph->param("W1", {NUM_FEATURES, 5}, init=inits::uniform());
  auto b1 = graph->param("b1", {1, 5}, init=inits::zeros);
  auto h = tanh(affine(x, W1, b1));

  // Define the output layer
  auto W2 = graph->param("W2", {5, NUM_LABELS}, init=inits::uniform());
  auto b2 = graph->param("b2", {1, NUM_LABELS}, init=inits::zeros);
  auto o = affine(h, W2, b2);

  if (train) {
    auto y = graph->constant({N, 1}, init=inits::from_vector(outputData));
    /* Define cross entropy cost on the output layer.
     *
     * It can be also defined directly as:
     *
     *   -mean(sum(logsoftmax(o) * y, axis=1), axis=0)
     *
     * But then `y` requires to be a one-hot-vector, i.e. [0,1,0, 1,0,0, 0,0,2,
     * ...] instead of [1, 0, 2, ...].
     */
    auto cost = mean(cross_entropy(o, y), axis=0);
    return cost;
  }
  else {
    auto preds = logsoftmax(o);
    return preds;
  }
}

// Helper functions
void readIrisData(const std::string fileName, std::vector<float>& features, std::vector<float>& labels);
void shuffleData(std::vector<float>& features, std::vector<float>& labels);
float calculateAccuracy(const std::vector<float> probs, const std::vector<float> labels);


int main(int argc, char** argv) {
  // Initialize global settings
  auto options = New<Config>(argc, argv, false);

  // Read data set (all 150 examples)
  std::vector<float> trainX;
  std::vector<float> trainY;
  readIrisData("../src/examples/iris/iris.data", trainX, trainY);

  // Split shuffled data into training data (120 examples) and test data (rest 30 examples)
  shuffleData(trainX, trainY);
  std::vector<float> testX(trainX.end() - 30*NUM_FEATURES, trainX.end());
  trainX.resize(120 * NUM_FEATURES);
  std::vector<float> testY(trainY.end() - 30, trainY.end());
  trainY.resize(120);

  {
    // Create network graph
    auto graph = New<ExpressionGraph>();

    // Set general options
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    // Choose optimizer (Sgd, Adagrad, Adam) and initial learning rate
    auto opt = Optimizer<Adam>(0.005);

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
      // Shuffle data in each epochs
      shuffleData(trainX, trainY);

      // Build classifier
      auto cost = buildIrisClassifier(graph, trainX, trainY, true);

      // Train classifier and update weights
      graph->forward();
      graph->backward();
      opt->update(graph);

      if (epoch % 10 == 0)
        std::cout << "Epoch: " << epoch << " Cost: " << cost->scalar() << std::endl;
    }

    // Build classifier with test data
    auto probs = buildIrisClassifier(graph, testX);

    // Print probabilities for debugging. The `debug` function has to be called
    // prior to computations in the network.
    //debug(probs, "Classifier probabilities")

    // Run classifier
    graph->forward();

    // Extract predictions
    std::vector<float> preds(testY.size());
    probs->val()->get(preds);

    std::cout << "Accuracy: " << calculateAccuracy(preds, testY) << std::endl;
  }

  return 0;
}


void readIrisData(const std::string fileName, std::vector<float>& features, std::vector<float>& labels) {
  std::map<std::string, int> CLASSES = {
    {"Iris-setosa", 0},
    {"Iris-versicolor", 1},
    {"Iris-virginica", 2}
  };

  std::ifstream in(fileName);
  if (! in.is_open()) {
    std::cerr << "Iris dataset not found: " << fileName << std::endl;
  }
  std::string line;
  std::string value;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    int i = 0;
    while (std::getline(ss, value, ',')) {
      if (++i == 5)
        labels.emplace_back(CLASSES[value]);
      else
        features.emplace_back(std::stof(value));
    }
  }
}

void shuffleData(std::vector<float>& features, std::vector<float>& labels) {
  // Create a list of indeces 0...K
  std::vector<int> indeces;
  indeces.reserve(labels.size());
  for (int i = 0; i < labels.size(); ++i)
    indeces.push_back(i);

  // Shuffle indeces
  std::random_shuffle(indeces.begin(), indeces.end());

  std::vector<float> featuresTemp;
  featuresTemp.reserve(features.size());
  std::vector<float> labelsTemp;
  labelsTemp.reserve(labels.size());

  // Get shuffled features and labels
  for (auto i = 0; i < indeces.size(); ++i) {
    auto idx = indeces[i];
    labelsTemp.push_back(labels[idx]);
    featuresTemp.insert(featuresTemp.end(),
                        features.begin() + (idx*NUM_FEATURES),
                        features.begin() + ((idx+1)*NUM_FEATURES));
  }

  features = featuresTemp;
  labels = labelsTemp;
}

float calculateAccuracy(const std::vector<float> probs, const std::vector<float> labels) {
  size_t numCorrect = 0;
  for (size_t i = 0; i < probs.size(); i += NUM_LABELS) {
    auto pred = std::distance(probs.begin() + i,
                              std::max_element(probs.begin() + i, probs.begin() + i + NUM_LABELS));
    if (pred == labels[i / NUM_LABELS])
      ++numCorrect;
  }
  return numCorrect / float(labels.size());
}
