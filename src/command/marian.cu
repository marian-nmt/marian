#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <boost/program_options.hpp>

#include "marian.h"
#include "optimizers/optimizers.h"
#include "optimizers/clippers.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/nematus.h"

#include "common/logging.h"
#include "command/config.h"

namespace marian {

  void TrainingLoop(Config& options,
                    Ptr<data::BatchGenerator<data::Corpus>> batchGenerator,
                    std::function<float(Ptr<data::CorpusBatch>)> update,
                    std::function<void(const std::string&)> save) {
    boost::timer::cpu_timer timer;

    size_t epochs = 1;
    size_t batches = 0;
    while((options["after-epochs"].as<size_t>() == 0
           || epochs <= options["after-epochs"].as<size_t>()) &&
          (options["after-batches"].as<size_t>() == 0
           || batches < options["after-batches"].as<size_t>())) {

      batchGenerator->prepare();

      float costSum = 0;
      size_t samples = 0;
      size_t wordsDisp = 0;

      while(*batchGenerator) {
        auto batch = batchGenerator->next();

        float cost = update(batch);

        costSum += cost;
        samples += batch->size();
        wordsDisp += batch->words();
        batches++;
        if(options["after-batches"].as<size_t>()
           && batches >= options["after-batches"].as<size_t>())
          break;

        if(batches % options["disp-freq"].as<size_t>() == 0) {
          std::stringstream ss;
          ss << "Ep. " << epochs
             << " : Up. " << batches
             << " : Sen. " << samples
             << " : Cost " << std::fixed << std::setprecision(2)
                           << costSum / options["disp-freq"].as<size_t>()
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

        if(batches % options["save-freq"].as<size_t>() == 0) {
          if(options["overwrite"].as<bool>())
            save(options["model"].as<std::string>() + ".npz");
          else
            save(options["model"].as<std::string>() + "." + std::to_string(batches) + ".npz");
        }
      }
      epochs++;
      LOG(info) << "Starting epoch " << epochs << " after " << samples << " samples";
    }
    LOG(info) << "Training finshed";
    save(options["model"].as<std::string>() + ".npz");
    LOG(info) << timer.format(2, "%ws");
  }
}

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;
  using namespace keywords;

  std::shared_ptr<spdlog::logger> info;
  info = spdlog::stderr_logger_mt("info");
  info->set_pattern("[%Y-%m-%d %T] %v");

  Config options(argc, argv);
  std::cerr << options << std::endl;

  auto dimVocabs = options["dim-vocabs"].as<std::vector<int>>();
  int dimEmb = options["dim-emb"].as<int>();
  int dimRnn = options["dim-rnn"].as<int>();
  int dimBatch = options["mini-batch"].as<int>();
  int dimMaxiBatch = options["maxi-batch"].as<int>();

  std::vector<int> dims = {
    dimVocabs[0], dimEmb, dimRnn,
    dimVocabs[1], dimEmb, dimRnn,
    dimBatch
  };

  int device = options["device"].as<int>();

  auto graph = New<ExpressionGraph>();
  graph->setDevice(device);

  auto nematus = New<Nematus>(dims);

  if(options.has("init")) {
    std::string modelInit = options.get<std::string>("init");
    LOG(info) << "Loading parameters from " << modelInit;
    nematus->load(graph, modelInit);
  }

  graph->reserveWorkspaceMB(options["workspace"].as<size_t>());

  Ptr<ClipperBase> clipper = nullptr;

  float clipNorm = options["clip-norm"].as<double>();
  float lrate = options["lrate"].as<double>();
  if(clipNorm > 0)
    clipper = Clipper<Norm>(clipNorm);
  auto opt = Optimizer<Adam>(lrate, clip=clipper);

  auto update = [graph, opt, nematus](Ptr<CorpusBatch> batch) -> float {
    auto costNode = nematus->construct(graph, batch);
    opt->update(graph);
    return costNode->scalar();
  };

  auto save = [graph, nematus](const std::string& name) {
    LOG(info) << "Saving parameters to " << name;
    nematus->save(graph, name);
  };


  auto trainSets = options["trainsets"].as<std::vector<std::string>>();
  auto vocabs = options["vocabs"].as<std::vector<std::string>>();
  size_t maxSentenceLength = options["max-length"].as<size_t>();

  auto corpus = New<Corpus>(trainSets, vocabs, dimVocabs, maxSentenceLength);
  auto bg = New<BatchGenerator<Corpus>>(corpus, dimBatch, dimMaxiBatch);

  TrainingLoop(options, bg, update, save);

  return 0;
}
