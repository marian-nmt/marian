#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

template <class Model>
class Train : public ModelTask {
private:
  Ptr<Config> options_;

public:
  Train(Ptr<Config> options) : options_(options) {}

  void run() {
    using namespace data;

    typedef typename Model::builder_type builder_type;
    typedef typename Model::dataset_type dataset_type;

    auto dataset = New<dataset_type>(options_);
    dataset->prepare();

    Ptr<BatchStats> stats;
    if(options_->get<bool>("dynamic-batching")) {
      LOG(info)->info("[batching] Collecting statistics for dynamic batching");
      // @TODO, better fake batch with vocabulary
      auto model = New<Model>(options_);
      THREAD_GUARD(stats = model->collectStats());
      LOG(info)->info("[batching] Done");
    }

    auto trainState = New<TrainingState>(options_);
    auto scheduler = New<Scheduler<dataset_type>>(options_, trainState);

    if((options_->has("valid-sets") || options_->has("valid-script-path"))
       && options_->get<size_t>("valid-freq") > 0) {
      for(auto validator :
          Validators<builder_type>(dataset->getVocabs(), options_))
        scheduler->addValidator(validator);
    }

    auto model = New<Model>(options_);
    model->setScheduler(scheduler);
    model->load();

    auto batchGenerator
        = New<BatchGenerator<dataset_type>>(dataset, options_, stats);

    scheduler->started();
    while(scheduler->keepGoing()) {
      auto shuffle = !options_->get<bool>("no-shuffle");
      batchGenerator->prepare(shuffle);
      while(*batchGenerator && scheduler->keepGoing()) {
        auto batch = batchGenerator->next();
        model->update(batch);
      }
      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();
    model->save(true);
  }
};
}
