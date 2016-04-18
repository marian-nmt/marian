#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include <atomic>
#include <boost/timer/timer.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>

#include "dl4mt.h"
#include "vocab.h"
#include "search.h"
#include "threadpool.h"

class BPE {
  public:
    BPE(const std::string& sep = "@@ ")
     : sep_(sep) {}
    
    std::string split(const std::string& line) {
      return line;
    }
    
    std::string unsplit(const std::string& line) {
      std::string joined = line;
      size_t pos = joined.find(sep_);
      while(pos != std::string::npos) {
        joined.erase(pos, sep_.size());
        pos = joined.find(sep_, pos);
      }
      return joined;
    }
    
    operator bool() const {
      return false;
    }
    
  private:
    std::string sep_;
};

int main(int argc, char* argv[]) {
  std::string srcVocabPath, trgVocabPath;
  std::vector<std::string> modelPaths;
  std::vector<std::string> lmPaths;
  std::vector<float> lmWeights;
  std::vector<size_t> devices;
  size_t nbest = 0;
  size_t beamSize = 12;
  size_t threads = 1;
  bool help = false;

  namespace po = boost::program_options;
  po::options_description cmdline_options("Allowed options");
  cmdline_options.add_options()
    ("beamsize,b", po::value(&beamSize)->default_value(12),
     "Beam size")
    ("threads", po::value(&threads)->default_value(1),
     "Number of threads")
    ("n-best-list", po::value(&nbest)->default_value(0),
     "N-best list")
    ("device(s),d", po::value(&devices)->multitoken(),
     "CUDA Device")
    ("model(s),m", po::value(&modelPaths)->multitoken()->required(),
     "Path to a model")
    ("lms(s),l", po::value(&lmPaths)->multitoken(),
     "Path to a kenlm language model")
    ("lw(s)", po::value(&lmWeights)->multitoken(),
     "Language Model weights")
    ("source,s", po::value(&srcVocabPath)->required(),
     "Path to a source vocab file.")
    ("target,t", po::value(&trgVocabPath)->required(),
     "Path to a target vocab file.")
    ("help,h", po::value(&help)->zero_tokens()->default_value(false),
     "Print this help message and exit.");
  
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).
              options(cmdline_options).run(), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl << std::endl;

    std::cout << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cout << cmdline_options << std::endl;
    exit(0);
  }

  if (help) {
    std::cout << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cout << cmdline_options << std::endl;
    exit(0);
  }

  if(devices.empty())
    devices.push_back(0);
  
  Vocab srcVocab(srcVocabPath);
  Vocab trgVocab(trgVocabPath);
  
  typedef std::unique_ptr<Weights> Model;
  typedef std::vector<Model> Models;
  typedef std::vector<Models> ModelsPerDevice;
  
  ModelsPerDevice modelsPerDevice(devices.size());
  
  {
    ThreadPool devicePool(devices.size());
    for(size_t i = 0; i < devices.size(); ++i) {
      std::cerr << "Loading model " << modelPaths[i] << " onto gpu" << devices[i] << std::endl;
      devicePool.enqueue([i, &devices, &modelsPerDevice, &modelPaths]{
        cudaSetDevice(devices[i]);
        for(auto& modelPath : modelPaths) {
          modelsPerDevice[i].emplace_back(new Weights(modelPath, devices[i]));
        }
      });
    }
  }
  
  std::vector<LM> lms;
  if(lmWeights.size() < lmPaths.size())
    lmWeights.resize(lmPaths.size(), 0.2);
  for(auto& lmPath : lmPaths) {
    std::cerr << "Loading lm " << lmPath << std::endl;
    size_t index = lms.size();
    float weight = lmWeights[index];
    lms.emplace_back(lmPath, trgVocab, index, weight);
  }
  
  std::cerr << "done." << std::endl;

  std::cerr << "Translating...\n";

  std::ios_base::sync_with_stdio(false);

  BPE bpe;
  
  
  boost::timer::cpu_timer timer;
  
  ThreadPool pool(std::max(threads, devices.size()));
  std::vector<std::future<History>> results;
  
  std::string in;
  std::size_t threadCounter = 0;
  while(std::getline(std::cin, in)) {
    auto call = [in, beamSize, threadCounter, devices, nbest, &modelsPerDevice, &lms, &bpe, &srcVocab] {
      thread_local std::unique_ptr<Search> search;
      if(!search) {
        Models& models = modelsPerDevice[threadCounter % devices.size()];
        cudaSetDevice(models[0]->GetDevice());
        search.reset(new Search(models, lms, nbest > 0));
      }
      
      Sentence sentence = bpe ? srcVocab(bpe.split(in)) : srcVocab(in);
      return search->Decode(sentence, beamSize);
    };
  
    results.emplace_back(
      pool.enqueue(call)
    );
    threadCounter++;
  }
  
  size_t lineCounter = 0;
  for(auto&& result : results) {
    History history = result.get();
    std::string out = trgVocab(history.Top().first);
    if(bpe)
      out = bpe.unsplit(out);
    std::cout << out << std::endl;
    if(nbest > 0) {
      NBestList nbl = history.NBest(nbest);
      for(size_t i = 0; i < nbl.size(); ++i) {
        auto& r = nbl[i];
        std::cout << lineCounter << " ||| " << (bpe ? bpe.unsplit(trgVocab(r.first)) : trgVocab(r.first, false)) << " |||";
        for(size_t j = 0; j < r.second->GetCostBreakdown().size(); ++j) {
          std::cout << " F" << j << "=" << r.second->GetCostBreakdown()[j];
        }
        std::cout << " ||| " << r.second->GetCost() << std::endl;
      }
    }
    history.Clear();
    lineCounter++;
  }
  std::cerr << timer.format() << std::endl;
  return 0;
}
