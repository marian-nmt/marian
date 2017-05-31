#pragma once

#include <iostream>
#include <iomanip>
#include <boost/timer/timer.hpp>

#include "common/keywords.h"
#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "optimizers/optimizers.h"

#include "examples/mnist/batch_generator.h"
#include "examples/mnist/feedforward.h"


using namespace data;
using namespace keywords;
using namespace models;

namespace marian {
namespace models {

class MNISTModel {
  private:
    Ptr<Config> options_;
    std::vector<int> dims_;
    bool inference_{false};

  public:
    template <class ...Args>
    MNISTModel(Ptr<Config> options, Args ...args)
      : options_(options),
    // FIXME
        dims_({784, 2028, 2048, 10}),
        inference_(Get(keywords::inference, false, args...))
    { }

    void load(Ptr<ExpressionGraph> graph, const std::string& name) {
      LOG(info, "Loading MNIST model...");
    }

    void save(Ptr<ExpressionGraph> graph, const std::string& name, bool foo=true) {
      LOG(info, "Saving MNIST model " + name);
    }

    Expr build(Ptr<ExpressionGraph> graph, Ptr<data::Batch> batch) {
      //LOG(info, "Building MNIST model...");
      return FeedforwardClassifier(graph, dims_ , batch, !inference_);
    }
};


}
}
