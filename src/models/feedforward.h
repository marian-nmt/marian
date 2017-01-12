#pragma once

#include "expression_graph.h"

namespace marian {

/**
 * @brief Namespace for code related to managing models in Marian
 */
namespace models {

/**
 * @brief Constructs an expression graph representing a feed-forward classifier.
 *
 * @param dims number of nodes in each layer of the feed-forward classifier
 *
 * @return a shared pointer to the newly constructed expression graph
 */
void FeedforwardClassifier(ExpressionGraphPtr g,
                           const std::vector<int>& dims,
                           size_t batchSize,
                           bool training = true) {
  using namespace keywords;
  std::cerr << "Building Multi-layer Feedforward network" << std::endl;
  std::cerr << "\tLayer dimensions:";
  for(auto d : dims)
    std::cerr << " " << d;
  std::cerr << std::endl;
  boost::timer::cpu_timer timer;

  // Construct a shared pointer to an empty expression graph
  g->clear();

  // Construct an input node called "x" and add it to the expression graph.
  //
  // For each observed data point, this input will hold a vector of values describing that data point.
  // dims.front() specifies the size of this vector
  //
  // For example, in the MNIST task, for any given image in the training set,
  //     "x" would hold a vector of pixel values for that image.
  //
  // Because calculating over one observed data point at a time can be inefficient,
  //     it is customary to operate over a batch of observed data points at once.
  //
  // At this point, we do not know the batch size:
  // whatevs therefore serves as a placeholder for the batch size, which will be specified later
  //
  // Once the batch size is known, "x" will represent a matrix with dimensions [batch_size, dims.front()].
  // Each row of this matrix will correspond with the observed data vector for one observed data point.
  auto x = name(g->input(shape={(int)batchSize, dims.front()}), "x");

  // Construct an input node called "y" and add it to the expression graph.
  //
  // For each observed data point, this input will hold the ground truth label for that data point.
  // dims.back() specifies the size of this vector
  //
  // For example, in the MNIST task, for any given image in the training set,
  //     "y" might hold one-hot vector representing which digit (0-9) is shown in that image
  //
  // Because calculating over one observed data point at a time can be inefficient,
  //     it is customary to operate over a batch of observed data points at once.
  //
  // At this point, we do not know the batch size:
  // whatevs therefore serves as a placeholder for the batch size, which will be specified later
  //
  // Once the batch size is known, "y" will represent a matrix with dimensions [batch_size, dims.front()].
  // Each row of this matrix will correspond with the ground truth data vector for one observed data point.
  auto y = name(g->input(shape={(int)batchSize, 1}), "y");

  std::vector<Expr> layers, weights, biases;
  for(int i = 0; i < dims.size()-1; ++i) {
    int in = dims[i];
    int out = dims[i+1];

    if(i == 0) {
      // Create a dropout node as the parent of x,
      //   and place that dropout node as the value of layers[0]
      layers.emplace_back(dropout(x, value=0.2));
    } else {
      // Multiply the matrix in layers[i-1] by the matrix in weights[i-1]
      // Take the result, and perform matrix addition on biases[i-1].
      // Wrap the result in rectified linear activation function,
      // and finally wrap that in a dropout node
      layers.emplace_back(dropout(relu(affine(layers.back(), weights.back(), biases.back())),
                                  value=0.5));
    }

    // Construct a weight node for the outgoing connections from layer i
    weights.emplace_back(
      g->param("W" + std::to_string(i), {in, out},
               init=inits::uniform()));

    // Construct a bias node. By definition, a bias node stores the value 1.
    //    Therefore, we don't actually store the 1.
    //    Instead, the bias node object stores the weights on the connections
    //      that are outgoing from the bias node.
    //    These weights are initialized to zero
    biases.emplace_back(
      g->param("b" + std::to_string(i), {1, out},
               init=inits::zeros));
  }

  // Perform matrix multiplication and addition for the last layer
  auto last = affine(layers.back(), weights.back(), biases.back());

  if(training) {
  // Define a top-level node for training
    auto cost = name(mean(cross_entropy(last, y), axis=0), "cost");
  }
  else {
    // Define a top-level node for inference
    auto scores = name(softmax(last), "scores");
  }

  std::cerr << "\tTotal time: " << timer.format(5, "%ws") << std::endl;
};

}
}
