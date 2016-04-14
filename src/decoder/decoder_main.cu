#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include <boost/timer/timer.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>

#include "dl4mt.h"
#include "vocab.h"
#include "search.h"

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
      return true;
    }
    
  private:
    std::string sep_;
};

int main(int argc, char* argv[]) {
  std::string srcVocabPath, trgVocabPath;
  std::vector<std::string> modelPaths;
  size_t device = 0;
  size_t nbest = 0;
  size_t beamSize = 12;
  bool help = false;

  namespace po = boost::program_options;
  po::options_description cmdline_options("Allowed options");
  cmdline_options.add_options()
    ("beamsize,b", po::value(&beamSize)->default_value(12),
     "Beam size")
    ("n-best-list", po::value(&nbest)->default_value(0),
     "N-best list")
    ("device,d", po::value(&device)->default_value(0),
     "CUDA Device")
    ("model(s),m", po::value(&modelPaths)->multitoken()->required(),
     "Path to a model")
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

  std::cerr << "Using device GPU" << device << std::endl;;
  cudaSetDevice(device);
  Vocab srcVocab(srcVocabPath);
  Vocab trgVocab(trgVocabPath);
  
  std::vector<std::unique_ptr<Weights>> models;
  for(auto& modelPath : modelPaths) {
    std::cerr << "Loading model " << modelPath << std::endl;
    models.emplace_back(new Weights(modelPath));
  }
  
  std::cerr << "done." << std::endl;

  Search search(models, nbest > 0);

  std::cerr << "Translating...\n";

  std::ios_base::sync_with_stdio(false);

  BPE bpe;
  
  boost::timer::cpu_timer timer;
  std::string in;
  size_t lineCounter = 0;
  while(std::getline(std::cin, in)) {
    Sentence sentence = bpe ? srcVocab(bpe.split(in)) : srcVocab(in);
    History history = search.Decode(sentence, beamSize);
    std::string out = trgVocab(history.Top().first);
    if(bpe)
      out = bpe.unsplit(out);
    std::cout << out << std::endl;
    if(nbest > 0) {
      NBestList nbl = history.NBest(beamSize);
      for(size_t i = 0; i < nbl.size(); ++i) {
        auto& r = nbl[i];
        std::cout << lineCounter << " ||| " << bpe.unsplit(trgVocab(r.first)) << " |||";
        for(size_t j = 0; j < r.second.GetCostBreakdown().size(); ++j) {
          std::cout << " F" << j << "=" << r.second.GetCostBreakdown()[j];
        }
        std::cout << " ||| " << r.second.GetCost() << std::endl;
      }
    }
    lineCounter++;
  }
  std::cerr << timer.format() << std::endl;
  return 0;
}
