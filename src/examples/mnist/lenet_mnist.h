#pragma once

#include <iomanip>
#include <iostream>
#include <memory>

#include <boost/timer/timer.hpp>

#include "common/definitions.h"
#include "common/keywords.h"
#include "graph/expression_graph.h"
#include "layers/convolution.h"

#include "examples/mnist/dataset.h"

namespace marian {
namespace models {

class LeNetModel {
private:
  Ptr<Config> options_;
  bool inference_{false};
  std::vector<int> dims_{784, 128, 10};

public:
  typedef data::MNIST dataset_type;

  template <class... Args>
  LeNetModel(Ptr<Config> options, Args... args)
      : options_(options),
        inference_(Get(keywords::inference, false, args...)) {}

  Expr build(Ptr<ExpressionGraph> graph, Ptr<data::Batch> batch) {
    return FeedforwardClassifier(graph, dims_, batch, inference_);
  }

  void load(Ptr<ExpressionGraph> graph, const std::string& name) {
    LOG(info)->critical("Loading MNIST model is not supported");
  }

  void save(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool foo = true) {
    LOG(info)->critical("Saving MNIST model is not supported");
  }

  Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph) {
    LOG(info)->critical("Collecting stats in MNIST model is not supported");
    return nullptr;
  }

private:
  /**
   * @brief Constructs an expression graph representing a feed-forward
   * classifier.
   *
   * @param dims number of nodes in each layer of the feed-forward classifier
   * @param batch a batch of training or testing examples
   * @param training create a classifier for training or for inference only
   *
   * @return a shared pointer to the newly constructed expression graph
   */
  Expr FeedforwardClassifier(Ptr<ExpressionGraph> g,
                             const std::vector<int>& dims,
                             Ptr<data::Batch> batch,
                             bool inference = false) {
    using namespace keywords;

    // Start with an empty expression graph
    g->clear();

    // Create an input layer of shape batchSize x numFeatures and populate it
    // with training features
    auto features
        = std::static_pointer_cast<data::DataBatch>(batch)->features();
    auto x = g->constant({(int)batch->size(), 1, 28, 28},
                         init = inits::from_vector(features));

    // Construct hidden layers

    auto conv_1 = Convolution("Conv1", 3, 3, 32)(x);
    auto conv_2 = relu(Convolution("Conv2", 3, 3, 64)(conv_1));
    auto pool = MaxPooling("MaxPooling", 2, 2)(conv_2);

    auto flatten = reshape(pool, {pool->shape()[0], pool->shape()[1] * pool->shape()[2] * pool->shape()[3], 1, 1});
    auto drop1 = dropout(flatten, keywords::dropout_prob=0.25);
    std::vector<Expr> layers, weights, biases;

    for(size_t i = 0; i < dims.size() - 1; ++i) {
      int in = dims[i];
      int out = dims[i + 1];

      if(i == 0) {
        // Create a dropout node as the parent of x,
        //   and place that dropout node as the value of layers[0]
        layers.emplace_back(drop1);
        in = drop1->shape()[1];
      } else {
        // Multiply the matrix in layers[i-1] by the matrix in weights[i-1]
        // Take the result, and perform matrix addition on biases[i-1].
        // Wrap the result in rectified linear activation function,
        // and finally wrap that in a dropout node
        layers.emplace_back(relu(affine(layers.back(), weights.back(), biases.back())));
      }

      // Construct a weight node for the outgoing connections from layer i
      weights.emplace_back(g->param(
          "W" + std::to_string(i), {in, out}, init = inits::uniform()));

      // Construct a bias node. These weights are initialized to zero
      biases.emplace_back(
          g->param("b" + std::to_string(i), {1, out}, init = inits::zeros));
    }

    // Perform matrix multiplication and addition for the last layer
    auto last = affine(dropout(layers.back(),keywords::dropout_prob=0.5), weights.back(), biases.back());
    // debug(last, "LAST");

    if(!inference) {
      // Create an output layer of shape batchSize x 1 and populate it with
      // labels
      auto labels = std::static_pointer_cast<data::DataBatch>(batch)->labels();
      auto y = g->constant({(int)batch->size(), 1},
                           init = inits::from_vector(labels));

      // debug(last, "last");

      // Define a top-level node for training
      return mean(cross_entropy(last, y), axis = 0);
    } else {
      // Define a top-level node for inference
      return logsoftmax(last);
    }
  };
};
}
}
