#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <boost/program_options.hpp>
#include <thread>
#include <chrono>
#include <mutex>

#include "marian.h"
#include "optimizers/optimizers.h"
#include "optimizers/clippers.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/nematus.h"

#include "common/logging.h"
#include "command/config.h"
#include "parallel/graph_group.h"

namespace marian {

  void TrainingLoop(Ptr<Config> options,
                    Ptr<data::BatchGenerator<data::Corpus>> batchGenerator) {

    auto reporter = New<Reporter>(options);
    Ptr<GraphGroup> graphGroup = New<AsyncGraphGroup<Nematus>>(options);
    graphGroup->setReporter(reporter);

    size_t epochs = 1;
    size_t batches = 0;
    while((options->get<size_t>("after-epochs") == 0
           || epochs <= options->get<size_t>("after-epochs")) &&
          (options->get<size_t>("after-batches") == 0
           || batches < options->get<size_t>("after-batches"))) {

      batchGenerator->prepare(!options->get<bool>("no-shuffle"));

      boost::timer::cpu_timer timer;

      while(*batchGenerator) {

        auto batch = batchGenerator->next();
        graphGroup->update(batch);

      }
      epochs++;
      LOG(info) << "Starting epoch " << epochs << " after "
        << reporter->samples << " samples";
    }
    LOG(info) << "Training finshed";
    graphGroup->save();
  }
}

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;
  using namespace keywords;

  Logger info{stderrLogger("info", "[%Y-%m-%d %T] %v")};
  Logger config{stderrLogger("config", "[config] %v")};
  Logger memory{stderrLogger("memory", "[memory] %v")};

  auto options = New<Config>(argc, argv);
  options->log();

  auto dimVocabs = options->get<std::vector<int>>("dim-vocabs");

  int dimBatch = options->get<int>("mini-batch");
  int dimMaxiBatch = options->get<int>("maxi-batch");
  
  auto corpus = New<Corpus>(options);
  auto bg = New<BatchGenerator<Corpus>>(corpus, dimBatch, dimMaxiBatch);

  TrainingLoop(options, bg);

  return 0;
}
