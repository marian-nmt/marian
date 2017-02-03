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
 
     
    auto save = [](const std::string& name) {
      LOG(info) << "Saving parameters to " << name;
      //nematus->save(graph, name);
    };

    auto reporter = New<Reporter>(options);
    auto graphGroup = New<SynchronousGraphGroup<Nematus>>(options);
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
        

        //    if(batches % options->get<size_t>("save-freq") == 0) {
        //      if(options->get<bool>("overwrite"))
        //        save(options->get<std::string>("model") + ".npz");
        //      else
        //        save(options->get<std::string>("model") + "." + std::to_string(batches) + ".npz");
        //    }
        
      }
      epochs++;
      LOG(info) << "Starting epoch " << epochs << " after "
        << reporter->samples << " samples";
    }
    LOG(info) << "Training finshed";
    save(options->get<std::string>("model") + ".npz");
  }
}

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;
  using namespace keywords;

  std::shared_ptr<spdlog::logger> info;
  info = spdlog::stderr_logger_mt("info");
  info->set_pattern("[%Y-%m-%d %T] %v");

  auto options = New<Config>(argc, argv);
  std::cerr << *options << std::endl;

  auto dimVocabs = options->get<std::vector<int>>("dim-vocabs");
  int dimEmb = options->get<int>("dim-emb");
  int dimRnn = options->get<int>("dim-rnn");
  int dimBatch = options->get<int>("mini-batch");
  int dimMaxiBatch = options->get<int>("maxi-batch");
  
  auto trainSets = options->get<std::vector<std::string>>("trainsets");
  auto vocabs = options->get<std::vector<std::string>>("vocabs");
  size_t maxSentenceLength = options->get<size_t>("max-length");
  auto corpus = New<Corpus>(trainSets, vocabs, dimVocabs, maxSentenceLength);
  auto bg = New<BatchGenerator<Corpus>>(corpus, dimBatch, dimMaxiBatch);
 
  TrainingLoop(options, bg);

  return 0;
}
