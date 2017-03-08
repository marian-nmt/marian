#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "training/config.h"
#include "training/validator.h"

namespace marian {

class Reporter {
  public:
    Ptr<Config> options_;
    std::vector<Ptr<Validator>> validators_;

    float costSum{0};
    size_t epochs{1};

    size_t samples{0};
    size_t wordsDisp{0};
    size_t batches{0};

    boost::timer::cpu_timer timer;

  public:
    Reporter(Ptr<Config> options) : options_(options) {}

    bool keepGoing() {
      // stop if it reached the maximum number of epochs
      if(options_->get<size_t>("after-epochs") > 0
         && epochs > options_->get<size_t>("after-epochs"))
        return false;

      // stop if it reached the maximum number of batch updates
      if(options_->get<size_t>("after-batches") > 0
         && batches >= options_->get<size_t>("after-batches"))
        return false;

      // stop if the first validator did not improve for a given number of checks
      if(options_->get<size_t>("early-stopping") > 0
         && !validators_.empty()
         && validators_[0]->stalled() >= options_->get<size_t>("early-stopping"))
        return false;

      return true;
    }

    void increaseEpoch() {
      LOG(info, "Seen {} samples", samples);

      epochs++;
      samples = 0;

      LOG(info, "Starting epoch {}", epochs);
    }

    void finished() {
      LOG(info, "Training finshed");
    }

    void addValidator(Ptr<Validator> validator) {
      validators_.push_back(validator);
    }

    void validate(Ptr<ExpressionGraph> graph) {
      if(batches % options_->get<size_t>("valid-freq") == 0) {
        for(auto validator : validators_) {
          if(validator) {
            size_t stalledPrev = validator->stalled();
            float value = validator->validate(graph);
            if(validator->stalled() > 0)
	      LOG(valid, "{} : {} : {} : stalled {} times", batches,
		  validator->type(), value, validator->stalled());
            else
	      LOG(valid, "{} : {} : {} : new best", batches,
		  validator->type(), value);
          }
        }
      }
    }

    size_t stalled() {
      for(auto validator : validators_)
        if(validator)
          return validator->stalled();
      return 0;
    }

    void update(float cost, Ptr<data::CorpusBatch> batch) {
      costSum += cost;
      samples += batch->size();
      wordsDisp += batch->words();
      batches++;

      if(batches % options_->get<size_t>("disp-freq") == 0) {
	LOG(info, "Ep. {} : Up. {} : Sen. : Cost {.2f} : Time {} : {.2f} words/s",
	    epochs, batches, samples, costSum / options_->get<size_t>("disp-freq"),
	    timer.format(2), wordsDisp / std::stof(timer.format(5, "%w")));
	timer.start();
        costSum = 0;
        wordsDisp = 0;
      }
    }
};

template <class Model>
void Train(Ptr<Config> options) {
  using namespace data;
  using namespace keywords;

  auto trainCorpus = New<Corpus>(options);
  auto batchGenerator = New<BatchGenerator<Corpus>>(trainCorpus,
                                                    options);
  auto reporter = New<Reporter>(options);

  if(options->has("valid-sets") && options->get<size_t>("valid-freq") > 0) {
    for(auto validator : Validators<typename Model::builder_type>(trainCorpus->getVocabs(),
                                                                  options))
      reporter->addValidator(validator);
  }

  auto model = New<Model>(options);
  model->setReporter(reporter);

  while(reporter->keepGoing()) {
    batchGenerator->prepare(!options->get<bool>("no-shuffle"));
    while(*batchGenerator && reporter->keepGoing()) {
      auto batch = batchGenerator->next();
      model->update(batch);
    }
    if(reporter->keepGoing())
      reporter->increaseEpoch();
  }
  reporter->finished();
  model->save();
}

}
