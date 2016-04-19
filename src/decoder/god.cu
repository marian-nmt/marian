#include <vector>

#include "god.h"
#include "threadpool.h"
#include "encoder_decoder.h"
#include "language_model.h"

God God::instance_;

God& God::Init(const std::string& initString) {
  std::vector<std::string> args = po::split_unix(initString);
  int argc = args.size() + 1;
  char* argv[argc];
  argv[0] = const_cast<char*>("dummy");
  for(int i = 1; i < argc; i++)
    argv[i] = const_cast<char*>(args[i-1].c_str());
  return Init(argc, argv);
}

God& God::Init(int argc, char** argv) {
  return Summon().NonStaticInit(argc, argv);
}

God& God::NonStaticInit(int argc, char** argv) {

  po::options_description general("General options");
  
  std::vector<size_t> devices;
  std::vector<std::string> modelPaths;
  std::vector<std::string> lmPaths;
  std::string sourceVocabPath;
  std::string targetVocabPath;
  
  general.add_options()
    ("model,m", po::value(&modelPaths)->multitoken()->required(),
     "Path to neural translation model(s)")
    ("source,s", po::value(&sourceVocabPath)->required(),
     "Path to source vocabulary file.")
    ("target,t", po::value(&targetVocabPath)->required(),
     "Path to target vocabulary file.")
    ("lm,l", po::value(&lmPaths)->multitoken(),
     "Paths to KenLM language model(s)")
    ("weights,w", po::value(&weights_)->multitoken(),
     "Model weights (for NMT and LM)")
    ("devices,d", po::value(&devices)->multitoken(),
     "Allowed CUDA Device(s)")
    ("threads", po::value<size_t>()->default_value(1),
     "Number of threads, at least equal to number of devices")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
     "Print this help message and exit")
  ;

  po::options_description search("Search options");
  search.add_options()
    ("beam-size,b", po::value<size_t>()->default_value(12),
     "Decoding beam-size")
    ("normalize,n", po::value<bool>()->zero_tokens()->default_value(false),
     "Normalize scores by translation length")
    ("n-best-list", po::value<bool>()->zero_tokens()->default_value(false),
     "Output n-best list with n = beam-size")
  ;

  po::options_description kenlm("KenLM specific options");
  kenlm.add_options()
    ("batch-size", po::value<size_t>()->default_value(1000),
     "Batch size for batched queries")
    ("batch-threads", po::value<size_t>()->default_value(4),
     "Concurrent worker threads for batch processing")
  ;

  po::options_description cmdline_options("Allowed options");
  cmdline_options.add(general);
  cmdline_options.add(search);
  cmdline_options.add(kenlm);

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

  if (Get<bool>("help")) {
    std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cerr << cmdline_options << std::endl;
    exit(0);
  }

  sourceVocab_.reset(new Vocab(sourceVocabPath));
  targetVocab_.reset(new Vocab(targetVocabPath));
  
  if(devices.empty())
    devices.push_back(0);
  
  modelsPerDevice_.resize(devices.size());
  
  {
    ThreadPool devicePool(devices.size());
    for(auto& modelPath : modelPaths) {
      for(size_t i = 0; i < devices.size(); ++i) {
        std::cerr << "Loading model " << modelPath << " onto gpu" << devices[i] << std::endl;
        devicePool.enqueue([i, &devices, &modelPath, this]{
          cudaSetDevice(devices[i]);
          modelsPerDevice_[i].emplace_back(new Weights(modelPath, devices[i]));
        });
      }
    }
  }
  
  for(auto& lmPath : lmPaths) {
    std::cerr << "Loading lm " << lmPath << std::endl;
    lms_.emplace_back(lmPath, *targetVocab_);
  }

  if(weights_.size() < modelPaths.size())
    weights_.resize(modelPaths.size(), 1.0);
  
  if(weights_.size() < lmPaths.size())
    weights_.resize(weights_.size() + lmPaths.size(), 0.0);
  
  std::cerr << "done." << std::endl;

  return *this;
}

Vocab& God::GetSourceVocab() {
  return *Summon().sourceVocab_;
}

Vocab& God::GetTargetVocab() {
  return *Summon().targetVocab_;
}

std::vector<ScorerPtr> God::GetScorers(size_t threadId) {
  size_t deviceId = threadId % Summon().modelsPerDevice_.size();
  cudaSetDevice(Summon().modelsPerDevice_[deviceId][0]->GetDevice());
  std::vector<ScorerPtr> scorers;
  for(auto& m : Summon().modelsPerDevice_[deviceId])
    scorers.emplace_back(new EncoderDecoder(*m));
  for(auto& lm : Summon().lms_)
    scorers.emplace_back(new LanguageModel(lm));
  return scorers;
}

std::vector<float>& God::GetScorerWeights() {
  return Summon().weights_;
}

// clean up cuda vectors before cuda context goes out of scope
void God::CleanUp() {
  for(auto& models : Summon().modelsPerDevice_)
    for(auto& m : models)
      m.reset(nullptr);
}


