#pragma once

#include <iomanip>
#include <iostream>
#include <memory>

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "models/costs.h"
#include "models/model_base.h"
#include "layers/loss.h"

#include "examples/mnist/dataset.h"

namespace marian {
namespace models {

// @TODO: looking at this file, simplify the new RationalLoss idea. Here it gets too complicated

class MNISTCrossEntropyCost : public CostBase {
public:
  MNISTCrossEntropyCost() {}

  Ptr<MultiRationalLoss> apply(Ptr<ModelBase> model,
                               Ptr<ExpressionGraph> graph,
                               Ptr<data::Batch> batch,
                               bool clearGraph = true) override {
    auto top = model->build(graph, batch, clearGraph);

    auto vfLabels = std::static_pointer_cast<data::DataBatch>(batch)->labels();

    // convert float to IndexType
    std::vector<IndexType> vLabels(vfLabels.begin(), vfLabels.end());
    auto labels = graph->indices(vLabels);

    // Define a top-level node for training
    // use CE loss

    auto loss = sum(cross_entropy(top->loss(), labels), /*axis =*/ 0);
    auto multiLoss = New<SumMultiRationalLoss>();
    multiLoss->push_back({loss, (float)vLabels.size()});
    return multiLoss;
  }
};

class MNISTLogsoftmax : public CostBase {
public:
  MNISTLogsoftmax() {}

  Ptr<MultiRationalLoss> apply(Ptr<ModelBase> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) override {
    auto top = model->build(graph, batch, clearGraph);
    
    // @TODO: simplify this
    auto multiLoss = New<SumMultiRationalLoss>();
    multiLoss->push_back({logsoftmax(top->loss()), top->count()});
    return multiLoss;
  }
};

class MnistFeedForwardNet : public ModelBase {
public:
  typedef data::MNISTData dataset_type;

  template <class... Args>
  MnistFeedForwardNet(Ptr<Options> options, Args... args)
      : options_(options), inference_(options->get<bool>("inference", false)) {}

  virtual Ptr<RationalLoss> build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool /*clean*/ = false) override {
    
    auto loss   = construct(graph, batch, inference_); // @TODO: unify nomenclature, e.g. rather use apply
    auto count = graph->constant({(int)batch->size(), 1}, inits::from_value(1.f));

    return New<RationalLoss>(loss, count);
  }

  void load(Ptr<ExpressionGraph> /*graph*/, const std::string& /*name*/, bool) override {
    LOG(critical, "Loading MNIST model is not supported");
  }

  void save(Ptr<ExpressionGraph> /*graph*/, const std::string& /*name*/, bool) override {
    LOG(critical, "Saving MNIST model is not supported");
  }

  void save(Ptr<ExpressionGraph> /*graph*/, const std::string& /*name*/) {
    LOG(critical, "Saving MNIST model is not supported");
  }

  Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> /*graph*/,
                                     size_t /*multiplier*/) {
    LOG(critical, "Collecting stats in MNIST model is not supported");
    return nullptr;
  }

  virtual void clear(Ptr<ExpressionGraph> graph) override { graph->clear(); };

protected:
  Ptr<Options> options_;
  bool inference_{false};

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
  virtual Expr construct(Ptr<ExpressionGraph> g,
                         Ptr<data::Batch> batch,
                         bool /*inference*/ = false) {
    const std::vector<int> dims = {784, 2048, 2048, 10};

    // Start with an empty expression graph
    clear(g);

    // Create an input layer of shape batchSize x numFeatures and populate it
    // with training features
    auto features
        = std::static_pointer_cast<data::DataBatch>(batch)->features();
    auto x = g->constant({(int)batch->size(), dims[0]},
                         inits::from_vector(features));

    // Construct hidden layers
    std::vector<Expr> layers, weights, biases;

    for(size_t i = 0; i < dims.size() - 1; ++i) {
      int in = dims[i];
      int out = dims[i + 1];

      if(i == 0) {
        // Create a dropout node as the parent of x,
        //   and place that dropout node as the value of layers[0]
        layers.emplace_back(dropout(x, 0.2));
      } else {
        // Multiply the matrix in layers[i-1] by the matrix in weights[i-1]
        // Take the result, and perform matrix addition on biases[i-1].
        // Wrap the result in rectified linear activation function,
        // and finally wrap that in a dropout node
        layers.emplace_back(dropout(
            relu(affine(layers.back(), weights.back(), biases.back())), 0.2));
      }

      // Construct a weight node for the outgoing connections from layer i
      weights.emplace_back(
          g->param("W" + std::to_string(i), {in, out}, inits::glorot_uniform));

      // Construct a bias node. These weights are initialized to zero
      biases.emplace_back(
          g->param("b" + std::to_string(i), {1, out}, inits::zeros));
    }

    // Perform matrix multiplication and addition for the last layer
    auto last = affine(layers.back(), weights.back(), biases.back());
    return last;
  }
};
}  // namespace models
}  // namespace marian
