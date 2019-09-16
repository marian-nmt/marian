#pragma once

#include "examples/mnist/model.h"
#include "layers/convolution.h"

namespace marian {
namespace models {

class MnistLeNet : public MnistFeedForwardNet {
public:
  template <class... Args>
  MnistLeNet(Ptr<Options> options, Args... args)
      : MnistFeedForwardNet(options, args...) {}

  virtual void clear(Ptr<ExpressionGraph> graph) override { graph->clear(); };

protected:
  virtual Expr apply(Ptr<ExpressionGraph> g,
                     Ptr<data::Batch> batch,
                     bool inference = false) override {
    const std::vector<int> dims = {784, 128, 10};

    // Start with an empty expression graph
    clear(g);

    // Create an input layer of shape batchSize x numFeatures and populate it
    // with training features
    auto features
        = std::static_pointer_cast<data::DataBatch>(batch)->features();
    auto x = g->constant({(int)batch->size(), 1, 28, 28},
                         inits::fromVector(features));

    // Construct hidden layers

    // clang-format off
    auto conv_1 = convolution(g)
                    ("prefix", "conv_1")
                    ("kernel-dims", std::make_pair(3,3))
                    ("kernel-num", 32)
                    .apply(x);

    auto conv_2 = convolution(g)
                    ("prefix", "conv_2")
                    ("kernel-dims", std::make_pair(3,3))
                    ("kernel-num", 64)
                    .apply(conv_1);
    // clang-format on

    auto relued = relu(conv_2);
    auto pool = max_pooling(relued, 2, 2, 1, 1, 1, 1);

    auto flatten
        = reshape(pool,
                  {pool->shape()[0],
                   pool->shape()[1] * pool->shape()[2] * pool->shape()[3]});
    auto drop1 = dropout(flatten, 0.25);
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
        layers.emplace_back(
            relu(affine(layers.back(), weights.back(), biases.back())));
      }

      // Construct a weight node for the outgoing connections from layer i
      weights.emplace_back(
          g->param("W" + std::to_string(i), {in, out}, inits::uniform()));

      // Construct a bias node. These weights are initialized to zero
      biases.emplace_back(
          g->param("b" + std::to_string(i), {1, out}, inits::zeros()));
    }

    // Perform matrix multiplication and addition for the last layer
    auto last
        = affine(dropout(layers.back(), 0.5), weights.back(), biases.back());

    if(!inference) {
      // Create an output layer of shape batchSize x 1 and populate it with
      // labels
      auto labels = std::static_pointer_cast<data::DataBatch>(batch)->labels();
      auto y = g->constant({(int)batch->size(), 1}, inits::fromVector(labels));

      // Define a top-level node for training
      return mean(cross_entropy(last, y), /*axis =*/ 0);
    } else {
      // Define a top-level node for inference
      return logsoftmax(last);
    }
  }
};
}  // namespace models
}  // namespace marian
