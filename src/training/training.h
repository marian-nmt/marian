#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/corpus_sqlite.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

template <class ModelWrapper>
class Train : public ModelTask {
private:
  Ptr<Config> options_;

public:
  Train(Ptr<Config> options) : options_(options) {}

  void run() {
    using namespace data;

    Ptr<CorpusBase> dataset;
    if(!options_->get<std::string>("sqlite").empty())
      dataset = New<CorpusSQLite>(options_);
    else
      dataset = New<Corpus>(options_);

    dataset->prepare();

    Ptr<BatchStats> stats;
    if(options_->get<bool>("mini-batch-fit")) {
      LOG(info,
          "[batching] Collecting statistics for batch fitting with step size "
          "{}",
          options_->get<size_t>("mini-batch-fit-step"));
      // @TODO, better fake batch with vocabulary
      auto model = New<ModelWrapper>(options_);
      THREAD_GUARD(stats = model->collectStats());
      LOG(info, "[batching] Done");
    }

    auto trainState = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, trainState);

    if((options_->has("valid-sets") || options_->has("valid-script-path"))
       && options_->get<size_t>("valid-freq") > 0) {
      for(auto validator : Validators(dataset->getVocabs(), options_))
        scheduler->addValidator(validator);
    }

    auto batchGenerator = New<CorpusBatchGenerator>(dataset, options_, stats);
    scheduler->registerTrainingObserver(batchGenerator);

    auto model = New<ModelWrapper>(options_);
    model->setScheduler(scheduler);
    model->load();

    // @TODO: shuffle_ as a private attribute in BG
    auto shuffle = !options_->get<bool>("no-shuffle");
    bool restored = !options_->get<bool>("no-restore-corpus")
                    && batchGenerator->restore(trainState, shuffle);

    scheduler->started();
    while(scheduler->keepGoing()) {
      if(!restored)
        batchGenerator->prepare(shuffle);
      restored = false;

      while(*batchGenerator && scheduler->keepGoing()) {
        auto batch = batchGenerator->next();
        model->update(batch);
      }

      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();

    model->finalize();
    if(!trainState->loaded)
      model->save(true);
  }
};
}
