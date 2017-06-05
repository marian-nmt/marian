#pragma once

#include "models/model_task.h"
#include "training/config.h"
#include "training/reporter.h"

#include "examples/mnist/dataset.h"
#include "examples/mnist/validator.h"

namespace marian {

template <class Model>
class TrainMNIST : public ModelTask {
private:
  Ptr<Config> options_;

public:
  TrainMNIST(Ptr<Config> options) : options_(options) {}

  void run() {
    using namespace data;

    // Prepare data set
    auto paths = options_->get<std::vector<std::string>>("train-sets");
    auto dataset = New<typename Model::dataset_type>(paths);
    auto batchGenerator
        = New<BatchGenerator<data::MNIST>>(dataset, options_, nullptr);

    // Prepare reporter with validators
    auto reporter = New<Reporter<typename Model::dataset_type>>(options_);
    auto validator
        = New<AccuracyValidator<typename Model::builder_type>>(options_);
    reporter->addValidator(validator);

    // Prepare model
    auto model = New<Model>(options_);
    model->setReporter(reporter);
    model->load();

    // Run training
    while(reporter->keepGoing()) {
      batchGenerator->prepare(!options_->get<bool>("no-shuffle"));
      while(*batchGenerator && reporter->keepGoing()) {
        auto batch = batchGenerator->next();
        model->update(batch);
      }
      if(reporter->keepGoing())
        reporter->increaseEpoch();
    }
    reporter->finished();
  }
};
}
