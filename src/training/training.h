#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/model_task.h"
#include "training/config.h"
#include "training/reporter.h"
#include "training/validator.h"

namespace marian {

template <class Model>
class Train : public ModelTask {
  public:
    Ptr<Config> options_;

  public:
    Train(Ptr<Config> options) : options_(options) {}

    void run() {
      using namespace data;

      auto trainCorpus = New<Corpus>(options_);
      if(options_->has("guided-alignment"))
        trainCorpus->setWordAlignment(options_->get<std::string>("guided-alignment"));

      Ptr<BatchStats> stats;
      if(options_->get<bool>("dynamic-batching")) {
        LOG(info, "[batching] Collecting statistics for dynamic batching");
        // @TODO, better fake batch with vocabulary
        auto model = New<Model>(options_);
        THREAD_GUARD(stats = model->collectStats());
        LOG(info, "[batching] Done");
      }

      auto batchGenerator = New<BatchGenerator<Corpus>>(trainCorpus, options_, stats);
      auto reporter = New<Reporter<data::Corpus>>(options_);

      if((options_->has("valid-sets") || options_->has("valid-script-path"))
         && options_->get<size_t>("valid-freq") > 0) {
        for(auto validator : Validators<typename Model::builder_type>(trainCorpus->getVocabs(),
                                                                      options_))
          reporter->addValidator(validator);
      }

      auto model = New<Model>(options_);
      model->setReporter(reporter);
      model->load();

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
      model->save(true);
    }
};

}
