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

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;
  using namespace keywords;

  cudaSetDevice(0);

  std::string modelPrefix;
  std::string modelInit;
  bool modelOverwrite;

  std::string srcCorpusPath;
  std::string trgCorpusPath;
  std::string srcVocabPath;
  std::string trgVocabPath;

  size_t maxEpochs;
  size_t maxBatches;
  size_t dispFreq;
  size_t saveFreq;
  size_t workSpace;

  size_t maxSentenceLength;
  size_t miniBatchSize;
  size_t maxiBatchSize;
  double lrate;
  double clipNorm;
  int dimSrcVoc, dimTrgVoc, dimSrcEmb,
    dimTrgEmb, dimEncState, dimDecState;


  namespace po = boost::program_options;
  po::options_description general("General options");

  general.add_options()
    ("model,m", po::value(&modelPrefix)->default_value("./model"),
     "Path prefix for model to be saved")
    ("init,i", po::value(&modelInit),
     "Load weights from  arg  before training")
    ("overwrite", po::value(&modelOverwrite)->default_value(false),
     "Overwrite model with following checkpoints")

    ("source-corpus,S", po::value(&srcCorpusPath)->required(),
     "Path to source language training corpus")
    ("target-corpus,T", po::value(&trgCorpusPath)->required(),
     "Path to target language training corpus")
    ("source-vocab,s", po::value(&srcVocabPath)->required(),
     "Path to source vocab file")
    ("target-vocab,t", po::value(&trgVocabPath)->required(),
     "Path to target vocab file")
    ("max-epochs,e", po::value(&maxEpochs)->default_value(0),
     "Maximum number of epochs, 0 is infinity")
    ("max-batches", po::value(&maxBatches)->default_value(0),
     "Maximum number of batch updates, 0 is infinity")
    ("disp-freq", po::value(&dispFreq)->default_value(100),
     "Display information every  arg  updates")
    ("save-freq", po::value(&saveFreq)->default_value(10000),
     "Save model file every  arg  updates")
    ("work-space,w", po::value(&workSpace)->default_value(4096),
     "Preallocate  arg  MB of work space")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
     "Print this help message and exit")
  ;

  po::options_description hyper("Hyper-parameters");
  hyper.add_options()
    ("max-length", po::value(&maxSentenceLength)->default_value(50),
     "Maximum length of a sentence in a training sentence pair")
    ("mini-batch,b", po::value(&miniBatchSize)->default_value(40),
     "Size of mini-batch used during update")
    ("maxi-batch", po::value(&maxiBatchSize)->default_value(20),
     "Number of batches to preload for length-based sorting")
    ("lrate,l", po::value(&lrate)->default_value(0.0001),
     "Learning rate for Adam algorithm")
    ("clip-norm,c", po::value(&clipNorm)->default_value(1.f),
     "Clip gradient norm to  arg  (0 to disable)")
    ("dim-src-vocab", po::value(&dimSrcVoc)->default_value(40000),
     "Size of source vocabulary")
    ("dim-trg-vocab", po::value(&dimTrgVoc)->default_value(40000),
     "Size of target vocabulary")
    ("dim-src-emb", po::value(&dimSrcEmb)->default_value(512), "Size of source embedding")
    ("dim-trg-emb", po::value(&dimTrgEmb)->default_value(512), "Size of target embedding")
    ("dim-enc-hidden", po::value(&dimEncState)->default_value(1024), "Size of encoder hidden state")
    ("dim-dec-hidden", po::value(&dimDecState)->default_value(1024), "Size of decoder hidden state")
  ;

  po::options_description cmdline_options("Allowed options");
  cmdline_options.add(general);
  cmdline_options.add(hyper);

  po::variables_map vm_;
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

  //**************************************************************************//

  std::vector<std::string> files =
    { srcCorpusPath, trgCorpusPath };

  std::vector<std::string> vocabs =
    { srcVocabPath, trgVocabPath };

  auto corpus = DataSet<Corpus>(files, vocabs, maxSentenceLength);
  BatchGenerator<Corpus> bg(corpus, miniBatchSize, maxiBatchSize);

  std::vector<int> dims = {
    dimSrcVoc, dimSrcEmb, dimEncState,
    dimTrgVoc, dimTrgEmb, dimDecState,
    (int)miniBatchSize
  };
  auto nematus = New<Nematus>(dims);
  if(modelInit.size())
    nematus->load(modelInit);
  nematus->reserveWorkspaceMB(workSpace);

  ClipperBasePtr clipper = nullptr;
  if(clipNorm > 0)
    clipper = Clipper<Norm>(clipNorm);
  auto opt = Optimizer<Adam>(lrate, clip=clipper);

  float sum = 0;
  float samples = 0;
  float words = 0;
  boost::timer::cpu_timer timer;

  size_t epochs = 0;
  size_t batches = 0;
  while((maxEpochs == 0 || epochs < maxEpochs) &&
        (maxBatches == 0 || batches < maxBatches)) {
    bg.prepare();
    while(bg) {
      auto batch = bg.next();

      nematus->construct(*batch);

      opt->update(nematus);

      float cost = nematus->cost();
      sum += cost;
      samples += batch->size();
      words += batch->words();
      batches++;
      if(maxBatches && batches >= maxBatches)
        break;

      if(batches % dispFreq == 0) {
        std::cout << std::setfill(' ')
                  << "Epoch " << epochs
                  << " Update " << batches
                  << " Cost "   << std::setw(7) << std::setprecision(6) << cost
                  << " Avg: " << sum / dispFreq
                  << " UD " << timer.format(2, "%ws");

        float seconds = std::stof(timer.format(5, "%w"));
        float sentences = samples / seconds;
        float wps = words / seconds;

        std::cout << " " << std::setw(6)
                  << std::setprecision(5)
                  << sentences
                  << " sent/s "
                  << wps
                  << " words/s" << std::endl;
        timer.start();
        sum = 0;
        samples = 0;
        words = 0;
      }

      if(batches % saveFreq == 0) {
        if(modelOverwrite)
          nematus->save(modelPrefix + ".npz");
        else
          nematus->save(modelPrefix + "." + std::to_string(batches) + ".npz");
      }
    }
  }
  std::cout << "Training finshed" << std::endl;
  nematus->save(modelPrefix + ".npz");
  std::cout << timer.format(2, "%ws") << std::endl;

  return 0;
}
