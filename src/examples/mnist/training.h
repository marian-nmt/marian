#pragma once

#include "common/config.h"
#include "models/model_task.h"
#include "training/scheduler.h"

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

    // Prepare scheduler with validators
    auto trainState = New<TrainingState>(options_);
    auto scheduler = New<Scheduler<typename Model::dataset_type>>(options_, trainState);
    auto validator
        = New<AccuracyValidator<typename Model::builder_type>>(options_);
    scheduler->addValidator(validator);

    // Prepare model
    auto model = New<Model>(options_);
    model->setScheduler(scheduler);
    model->load();

    // Run training
    while(scheduler->keepGoing()) {
      batchGenerator->prepare(!options_->get<bool>("no-shuffle"));
      while(*batchGenerator && scheduler->keepGoing()) {
        auto batch = batchGenerator->next();
        model->update(batch);
      }
      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();
  }
};
}
