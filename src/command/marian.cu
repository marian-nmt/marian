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

namespace marian {

  void Config(int argc, char** argv,
              boost::program_options::variables_map& vm_) {

    namespace po = boost::program_options;
    po::options_description general("General options");

    general.add_options()
      ("model,m", po::value<std::string>()->default_value("./model"),
       "Path prefix for model to be saved")
      ("device,d", po::value<int>()->default_value(0),
       "Use device no.  arg")
      ("init,i", po::value<std::string>()->default_value(""),
       "Load weights from  arg  before training")
      ("overwrite", po::value<bool>()->default_value(false),
       "Overwrite model with following checkpoints")
      ("trainsets,t", po::value<std::vector<std::string>>()->multitoken()->required(),
       "Paths to training corpora")
      ("vocabs,v", po::value<std::vector<std::string>>()->multitoken()->required(),
       "Paths to vocabulary files, have to correspond with --trainsets")
      ("max-epochs,e", po::value<size_t>()->default_value(0),
       "Maximum number of epochs, 0 is infinity")
      ("max-batches", po::value<size_t>()->default_value(0),
       "Maximum number of batch updates, 0 is infinity")
      ("disp-freq", po::value<size_t>()->default_value(100),
       "Display information every  arg  updates")
      ("save-freq", po::value<size_t>()->default_value(30000),
       "Save model file every  arg  updates")
      ("workspace,w", po::value<size_t>()->default_value(2048),
       "Preallocate  arg  MB of work space")
      ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
       "Print this help message and exit")
    ;

    po::options_description hyper("Hyper-parameters");
    hyper.add_options()
      ("max-length", po::value<size_t>()->default_value(50),
       "Maximum length of a sentence in a training sentence pair")
      ("mini-batch,b", po::value<int>()->default_value(40),
       "Size of mini-batch used during update")
      ("maxi-batch", po::value<int>()->default_value(20),
       "Number of batches to preload for length-based sorting")
      ("lrate,l", po::value<float>()->default_value(0.0001),
       "Learning rate for Adam algorithm")
      ("clip-norm,c", po::value<float>()->default_value(1.f),
       "Clip gradient norm to  arg  (0 to disable)")
      ("dim-vocabs", po::value<std::vector<int>>()
        ->multitoken()
        ->default_value(std::vector<int>({50000, 50000}), "50000 50000"),
       "Maximum items in vocabulary ordered by rank")
      ("dim-emb", po::value<int>()->default_value(512), "Size of embedding vector")
      ("dim-rnn", po::value<int>()->default_value(1024), "Size of rnn hidden state")
    ;

    po::options_description cmdline_options("Allowed options");
    cmdline_options.add(general);
    cmdline_options.add(hyper);

    try {
      po::store(po::command_line_parser(argc,argv)
                .options(cmdline_options).run(), vm_);
      po::notify(vm_);
    }
    catch (std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl << std::endl;

      std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
      std::cerr << cmdline_options << std::endl;
      exit(1);
    }

    if (vm_["help"].as<bool>()) {
      std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
      std::cerr << cmdline_options << std::endl;
      exit(0);
    }
  }

  void TrainingLoop(boost::program_options::variables_map& options,
                    Ptr<data::BatchGenerator<data::Corpus>> batchGenerator,
                    std::function<float(Ptr<data::CorpusBatch>)> update,
                    std::function<void(const std::string&)> save) {
    boost::timer::cpu_timer timer;

    size_t epochs = 1;
    size_t batches = 0;
    while((options["max-epochs"].as<size_t>() == 0
           || epochs <= options["max-epochs"].as<size_t>()) &&
          (options["max-batches"].as<size_t>() == 0
           || batches < options["max-batches"].as<size_t>())) {

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
        if(options["max-batches"].as<size_t>()
           && batches >= options["max-batches"].as<size_t>())
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

  boost::program_options::variables_map vm_;
  Config(argc, argv, vm_);

  auto dimVocabs = vm_["dim-vocabs"].as<std::vector<int>>();
  int dimEmb = vm_["dim-emb"].as<int>();
  int dimRnn = vm_["dim-rnn"].as<int>();
  int dimBatch = vm_["mini-batch"].as<int>();
  int dimMaxiBatch = vm_["maxi-batch"].as<int>();

  std::vector<int> dims = {
    dimVocabs[0], dimEmb, dimRnn,
    dimVocabs[1], dimEmb, dimRnn,
    dimBatch
  };

  int device = vm_["device"].as<int>();

  auto graph = New<ExpressionGraph>();
  graph->setDevice(device);

  auto nematus = New<Nematus>(dims);

  std::string modelInit = vm_["init"].as<std::string>();
  if(modelInit.size()) {
    LOG(info) << "Loading parameters from " << modelInit;
    nematus->load(graph, modelInit);
  }

  graph->reserveWorkspaceMB(vm_["workspace"].as<size_t>());

  Ptr<ClipperBase> clipper = nullptr;

  float clipNorm = vm_["clip-norm"].as<float>();
  float lrate = vm_["lrate"].as<float>();

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


  auto trainSets = vm_["trainsets"].as<std::vector<std::string>>();
  auto vocabs = vm_["vocabs"].as<std::vector<std::string>>();
  size_t maxSentenceLength = vm_["max-length"].as<size_t>();

  auto corpus = New<Corpus>(trainSets, vocabs, dimVocabs, maxSentenceLength);
  auto bg = New<BatchGenerator<Corpus>>(corpus, dimBatch, dimMaxiBatch);
  TrainingLoop(vm_, bg, update, save);

  return 0;
}
