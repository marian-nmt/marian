#include <boost/utility/result_of.hpp>

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

void ProgramOptions(int argc, char *argv[],
    std::string& modelPath,
    std::string& svPath,
    std::string& tvPath,
    size_t& beamsize,
    size_t& device) {
  bool help = false;

  namespace po = boost::program_options;
  po::options_description cmdline_options("Allowed options");
  cmdline_options.add_options()
    ("beamsize,b", po::value(&beamsize)->default_value(12),
     "Beam size")
    ("device,d", po::value(&device)->default_value(0),
     "CUDA Device")
    ("model,m", po::value(&modelPath)->required(),
     "Path to a model")
    ("source,s", po::value(&svPath)->required(),
     "Path to a source vocab file.")
    ("target,t", po::value(&tvPath)->required(),
     "Path to a target vocab file.")
    ("help,h", po::value(&help)->zero_tokens()->default_value(false),
     "Print this help message and exit.")
  ;
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
}

int main(int argc, char* argv[]) {
  std::string modelPath, srcVocabPath, trgVocabPath;
  size_t device = 0;
  size_t beamSize = 12;
  ProgramOptions(argc, argv, modelPath, srcVocabPath, trgVocabPath, beamSize, device);
  std::cerr << "Using device GPU" << device << std::endl;;
  cudaSetDevice(device);
  std::cerr << "Loading model... ";
  Weights model(modelPath);
  Vocab srcVocab(srcVocabPath);
  Vocab trgVocab(trgVocabPath);
  std::cerr << "done." << std::endl;

  Search search(model, srcVocab, trgVocab);

  std::cerr << "Translating...\n";

  std::ios_base::sync_with_stdio(false);

  std::string line;
  boost::timer::cpu_timer timer;
  while(std::getline(std::cin, line)) {
    auto result = search.Decode(line, beamSize);
    std::cout << result << std::endl;
  }
  std::cerr << timer.format() << std::endl;
  return 0;
}
