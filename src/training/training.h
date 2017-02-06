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
      return
        (options_->get<size_t>("after-epochs") == 0
         || epochs <= options_->get<size_t>("after-epochs"))
        &&
        (options_->get<size_t>("after-batches") == 0
         || batches < options_->get<size_t>("after-batches"));
    }

    void increaseEpoch() {
      LOG(info) << "Seen " << samples << " samples";

      epochs++;
      samples = 0;

      LOG(info) << "Starting epoch " << epochs;
    }

    void finished() {
      LOG(info) << "Training finshed";
    }

    void addValidator(Ptr<Validator> validator) {
      validators_.push_back(validator);
    }
    
    void validate(Ptr<ExpressionGraph> graph) {
      if(batches % options_->get<size_t>("valid-freq") == 0) {
        LOG(valid) << "Validating after " << batches << " batches";
        for(auto validator : validators_) {
          if(validator) {
            float value = validator->validate(graph);
            LOG(valid) << validator->type() << " : " << value;
          }
        }
      }
    }
    
    void update(float cost, Ptr<data::CorpusBatch> batch) {
      static std::mutex sMutex;
      std::lock_guard<std::mutex> guard(sMutex);

      costSum += cost;
      samples += batch->size();
      wordsDisp += batch->words();
      batches++;

      if(batches % options_->get<size_t>("disp-freq") == 0) {
        std::stringstream ss;
        ss << "Ep. " << epochs
           << " : Up. " << batches
           << " : Sen. " << samples
           << " : Cost " << std::fixed << std::setprecision(2)
                         << costSum / options_->get<size_t>("disp-freq")
           << " : Time " << timer.format(2, "%ws");

        float seconds = std::stof(timer.format(5, "%w"));
        float wps = wordsDisp /   (float)seconds;

        ss << " : " << std::fixed << std::setprecision(2)
           << wps << " words/s";

        LOG(info) << ss.str();

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

  auto model = New<Model>(options);
  
  auto trainCorpus = New<Corpus>(options);
  auto batchGenerator = New<BatchGenerator<Corpus>>(trainCorpus,
                                                    options);
  auto reporter = New<Reporter>(options);
  
  if(options->has("valid-sets") && options->get<size_t>("valid-freq") > 0) {
    for(auto validator : Validators<typename Model::builder_type>(trainCorpus->getVocabs(),
                                                                  options))
      reporter->addValidator(validator);
  }
  
  model->setReporter(reporter);
  
  while(reporter->keepGoing()) {
    batchGenerator->prepare(!options->get<bool>("no-shuffle"));
    while(*batchGenerator && reporter->keepGoing()) {
      auto batch = batchGenerator->next();
      model->update(batch);
    }
    reporter->increaseEpoch();
  }
  reporter->finished();
  model->save();
}

}
