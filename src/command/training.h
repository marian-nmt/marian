#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "command/config.h"

namespace marian {

class Reporter {
  public:
    Ptr<Config> options_;

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

  auto reporter = New<Reporter>(options);
  auto model = New<Model>(options);

  model->setReporter(reporter);

  auto corpus = New<Corpus>(options);
  auto batchGenerator = New<BatchGenerator<Corpus>>(corpus, options);
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
