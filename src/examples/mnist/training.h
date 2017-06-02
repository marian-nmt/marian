#pragma once

#include "models/model_task.h"
#include "training/config.h"
#include "training/reporter.h"

#include "examples/mnist/mnist.h"
#include "examples/mnist/validator.h"


namespace marian {

template <class Model>
class MNISTTrain : public ModelTask {
  public:
    Ptr<Config> options_;

  public:
    MNISTTrain(Ptr<Config> options) : options_(options) {}

    void run() {
      using namespace data;

      auto paths = options_->get<std::vector<std::string>>("train-sets");
      auto dataset = New<MNIST>(paths);

      //Ptr<BatchStats> stats;
      //if(options_->get<bool>("dynamic-batching")) {
        //LOG(info, "[batching] Collecting statistics for dynamic batching");
        //// @TODO, better fake batch with vocabulary
        //auto model = New<Model>(options_);
        //THREAD_GUARD(stats = model->collectStats());
        //LOG(info, "[batching] Done");
      //}

      auto reporter = New<Reporter<MNIST>>(options_);
      std::vector<Ptr<Vocab>> fakeVocabs;
      auto validator
        = New<AccuracyValidator<typename Model::builder_type>>(fakeVocabs, options_);
      reporter->addValidator(validator);

      auto model = New<Model>(options_);
      model->setReporter(reporter);
      model->load();

      auto batchGenerator = New<BatchGenerator<MNIST>>(dataset, options_, nullptr);

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
      //model->save(true);
    }
};

}
