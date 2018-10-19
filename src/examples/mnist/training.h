#pragma once

#include "common/config.h"
#include "models/model_task.h"
#include "training/scheduler.h"

#include "examples/mnist/dataset.h"
#include "examples/mnist/validator.h"

namespace marian {

template <class ModelWrapper>
class TrainMNIST : public ModelTask {
private:
  Ptr<Config> options_;

public:
  TrainMNIST(Ptr<Config> options) : options_(options) {}

  void run() override {
    using namespace data;

    // @TODO: unify this and get rid of Config object.
    auto tOptions = New<Options>();
    tOptions->merge(options_);

    // Prepare data set
    auto paths = options_->get<std::vector<std::string>>("train-sets");
    auto dataset = New<data::MNISTData>(paths);
    auto batchGenerator = New<BatchGenerator<data::MNISTData>>(dataset, tOptions, nullptr);

    // Prepare scheduler with validators
    auto trainState = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, trainState);
    scheduler->addValidator(New<AccuracyValidator>(tOptions));

    // Prepare model
    auto model = New<ModelWrapper>(options_);
    model->setScheduler(scheduler);
    model->load();

    // Run training
    while(scheduler->keepGoing()) {
      batchGenerator->prepare(!options_->get<bool>("no-shuffle"));
      for(auto batch : *batchGenerator) {
        if(!scheduler->keepGoing())
           break;
        model->update(batch);
      }
      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();
  }
};
}  // namespace marian
